"""
Adapted and Extended from:

@inproceedings{shelmanov2021certain,
    title={How certain is your Transformer?},
    author={Shelmanov, Artem and Tsymbalov, Evgenii and Puzyrev, Dmitri and Fedyanin, Kirill and Panchenko, Alexander and Panov, Maxim},
    booktitle={Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume},
    pages={1833--1840},
    year={2021}
}

@inproceedings{roberts2024using,
    title={Using Artificial Populations to Study Psychological Phenomena in Neural Models},
    author={Roberts, Jesse and Moore, Kyle and Wilenzick, Drew and Fisher, Douglas},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    pages={18906-18914},
    year={2024}
}

PopulationLM
============
Utilities to approximate a population of individuals (language models) from a
single base transformer, so that psycholinguistic analysis can be conducted over
a distribution of behaviours rather than a single deterministic output.

This module exposes two tiers:

* Primitives (unchanged in spirit, kept for back-compat):
    ``DropoutMC`` and subclasses, ``DropoutUtils``,
    ``generate_dropout_population``, ``call_function_with_population``,
    ``generate_population_and_apply``.

* A convenience layer that orchestrates the primitives idempotently and makes a
  population a first-class, serialisable object:
    ``Population`` (save / load / validate),
    ``prepare_model`` (idempotent convert + insert + activate),
    ``deactivate`` / ``deactivated``,
    ``generate_population``,
    ``apply_population`` (validates, forces eval, controls decoding RNG).

Key design points
-----------------
* Stratified masks are *feature-only* (shape ``[1, ..., 1, H]``): they broadcast
  over any batch size and sequence length, so a population is independent of the
  data it is run over and is therefore reusable across batches, savable, and safe
  for free / multi-forward generation (chain-of-thought, ``model.generate``,
  beam search).
* A member's mask defines a single coherent sub-network ("subject") that is held
  fixed across every stimulus and every generation step.
"""


import os
import torch
from torch.nn import Identity
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable, Union, Dict, List, Callable, Optional, Any
from functools import partial


# ---------------------------------------------------------------------------
# Dropout modules
# ---------------------------------------------------------------------------
class DropoutMC(torch.nn.Module):
    def __init__(self, p: float, activate=False):
        super().__init__()
        self.activate = activate
        self.p = p
        self.p_init = p

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.dropout(
            x, self.p, training=self.training or self.activate
        )


class StratifiedDropoutMC(DropoutMC):
    """
    Random dropout that is generated once (the "identity") and then held fixed,
    so each generated identity represents a single individual in the population.

    The mask is *feature-only*: every axis except the last (feature/unit) axis is
    a singleton, so the same identity broadcasts to any batch size and sequence
    length. This makes a population independent of the data it is applied to and
    consistent across every forward pass of a free-generation decode.
    """
    def __init__(self, p: float, activate=False, batch_first: bool = True):
        super().__init__(p, activate)
        self.batch_first = batch_first
        self.identity = None

    def reset_identity(self):
        self.identity = None

    def forward(self, x: torch.Tensor):
        x = x.clone()

        if self.identity is None:
            # Feature-only mask: collapse every axis except the last to a
            # singleton so the mask broadcasts over batch and sequence. This also
            # avoids the previous behaviour of collapsing the feature axis on 2D
            # ([B, H]) tensors.
            size = [1] * (x.dim() - 1) + [x.size(-1)]
            m = x.data.new(torch.Size(size)).bernoulli_(1 - self.p)
            self.identity = m.div_(1 - self.p)

        if not self.activate or not self.p:
            return x

        # Vectorised single-pass mode (set transiently by PopulationModel): the
        # incoming batch has been tiled to ``groups * B`` in member-major order,
        # and ``_vector_identity`` holds the whole population's masks stacked on a
        # leading axis (shape ``[groups, 1, ..., 1, H]``). Reshape the leading
        # axis back to ``[groups, B, ...]`` so each member-block sees its own
        # mask, then flatten again. Valid because every op in a decoder is
        # independent across the batch axis, so this equals ``groups`` separate
        # forwards. See PopulationModel.forward(..., vectorized=True).
        vid = getattr(self, "_vector_identity", None)
        if vid is not None:
            groups = self._vector_groups
            per = x.shape[0] // groups
            xv = x.view(groups, per, *x.shape[1:])
            return (xv * vid).reshape(x.shape)

        return self.identity.expand_as(x) * x


class LockedDropoutMC(DropoutMC):
    """
    Implementation of locked (or variational) dropout. Randomly drops out entire parameters in embedding space.
    """

    def __init__(self, p: float, activate: bool = False, batch_first: bool = True):
        super().__init__(p, activate)
        self.batch_first = batch_first

    def forward(self, x):
        x = x.clone()
        if self.training:
            self.activate = True
        if not self.activate or not self.p:
            return x

        if not self.batch_first:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.p)
        else:
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)

        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.p)
        mask = mask.expand_as(x)
        return mask * x


class WordDropoutMC(DropoutMC):
    """
    Implementation of word dropout. Randomly drops out entire words (or characters) in embedding space.
    """

    def forward(self, x):
        if self.training:
            self.activate = True

        if not self.activate or not self.p:
            return x

        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - self.p)

        mask = torch.autograd.Variable(m, requires_grad=False)
        return mask * x


MC_DROPOUT_SUBSTITUTES = {
    "Dropout": DropoutMC,
    "LockedDropout": LockedDropoutMC,
    "WordDropout": WordDropoutMC,
    "StratifiedDropout": StratifiedDropoutMC,
}


# ---------------------------------------------------------------------------
# Insertion-point selection (generalised, structural, configurable)
# ---------------------------------------------------------------------------
_MLP_PATH_TOKENS = ("mlp", "ffn", "feed_forward", "feedforward")
_ATTN_PATH_TOKENS = ("attn", "attention")
# Used only when hidden_size cannot be discovered, to avoid wrapping the
# up/gate projections of an MLP (which project to intermediate_size).
_OUTPUT_NAME_HINTS = ("down_proj", "out_proj", "c_proj", "fc2", "dense_4h_to_h",
                      "wo", "w2", "output_dense")


def default_insertion_predicate(model: torch.nn.Module,
                                hidden_size: Optional[int] = None) -> Callable[[str, torch.nn.Module], bool]:
    """
    Build a structural predicate that selects the feed-forward output projection
    of each transformer block, without relying on architecture-specific class
    names. Targets ``torch.nn.Linear`` modules that:
      * live under an MLP/FFN path token (and not an attention path token), and
      * project back to ``hidden_size`` (the FFN down-projection).

    When ``hidden_size`` is unavailable, falls back to output-projection name
    hints. For architectures the heuristic misses, pass an explicit ``where``
    predicate to :func:`prepare_model` / :func:`generate_population`.
    """
    if hidden_size is None:
        cfg = getattr(model, "config", None)
        hidden_size = getattr(cfg, "hidden_size", None) if cfg is not None else None

    def predicate(name: str, module: torch.nn.Module) -> bool:
        if not isinstance(module, torch.nn.Linear):
            return False
        lname = name.lower()
        if any(t in lname for t in _ATTN_PATH_TOKENS):
            return False
        if not any(t in lname for t in _MLP_PATH_TOKENS):
            return False
        if hidden_size is not None:
            return module.out_features == hidden_size
        return any(h in lname for h in _OUTPUT_NAME_HINTS)

    return predicate


# ---------------------------------------------------------------------------
# DropoutUtils
# ---------------------------------------------------------------------------
class DropoutUtils():
    @classmethod
    def add_new_dropout_layers(
        cls,
        model: torch.nn.Module,
        where: Optional[Callable[[str, torch.nn.Module], bool]] = None,
        dropout_cls: type = StratifiedDropoutMC,
        p: float = 0.1,
        hidden_size: Optional[int] = None,
        verbose: bool = False,
    ) -> int:
        """
        Insert dropout into models that ship without usable dropout layers.

        Selection is structural and configurable via ``where`` (a predicate
        ``(qualified_name, module) -> bool``); the default targets the FFN
        down-projection of each block (see :func:`default_insertion_predicate`).
        Each selected linear ``L`` is replaced in its parent by
        ``Sequential(L, dropout_cls(p))``. Idempotent: a linear already wrapped
        with a trailing ``DropoutMC`` is skipped, so re-running never double-wraps.

        Returns the number of dropout layers inserted.
        """
        if where is None:
            where = default_insertion_predicate(model, hidden_size=hidden_size)

        count = 0
        # Snapshot the tree first so mutation during the walk is safe and newly
        # inserted modules are not revisited.
        for parent_name, parent in list(model.named_modules()):
            for child_name, child in list(parent.named_children()):
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name

                # Idempotency guard: already wrapped with an MC dropout.
                if (isinstance(child, torch.nn.Sequential)
                        and len(child) >= 1
                        and isinstance(child[-1], DropoutMC)):
                    continue

                if where(full_name, child):
                    new = torch.nn.Sequential(child, dropout_cls(p=p, activate=False))
                    setattr(parent, child_name, new)
                    count += 1
                    if verbose:
                        print(f"inserted dropout after: {full_name}")
        return count

    @classmethod
    def show_model(cls, model: torch.nn.Module):
        print([child for child in model.children()])

    @classmethod
    def _convert_to_mc_dropout(
        cls,
        model: torch.nn.Module,
        substitution_dict: Dict[str, Callable] = None,
        ignore_module_list: Iterable[str] = ('attn_dropout', 'attention'),
    ) -> int:
        layer_replaced_count = 0

        for i, layer in enumerate(list(model.children())):
            proba_field_name = "dropout_rate" if "flair" in str(type(layer)) else "p"
            module_name = list(model._modules.items())[i][0]
            layer_name = layer._get_name()
            if module_name in ignore_module_list:
                continue
            elif layer_name in substitution_dict.keys():
                model._modules[module_name] = substitution_dict[layer_name](
                    p=getattr(layer, proba_field_name), activate=False
                )
                layer_replaced_count += 1
            else:
                layer_replaced_count += cls._convert_to_mc_dropout(
                    model=layer, substitution_dict=substitution_dict,
                    ignore_module_list=ignore_module_list,
                )

        return layer_replaced_count

    @classmethod
    def activate_mc_dropout(
        cls, model: torch.nn.Module, activate: bool, random: float = 0.0, verbose: bool = False
    ):
        for layer in model.children():
            if isinstance(layer, DropoutMC):
                if verbose:
                    print(layer)
                    print(f"Current DO state: {layer.activate}")
                    print(f"Switching state to: {activate}")
                layer.activate = activate
                if activate:
                    layer.p = random
                if not activate:
                    layer.p = layer.p_init
            else:
                cls.activate_mc_dropout(
                    model=layer, activate=activate, random=random, verbose=verbose
                )

    @classmethod
    def reset_stratified_mc_dropout(cls, model: torch.nn.Module):
        for layer in model.children():
            if isinstance(layer, StratifiedDropoutMC):
                layer.reset_identity()
            else:
                cls.reset_stratified_mc_dropout(model=layer)

    @classmethod
    def get_stratified_dropout_identity(cls, model: torch.nn.Module):
        probs = {}
        identity = {}
        for name, layer in model.named_modules():
            if isinstance(layer, StratifiedDropoutMC):
                identity[name] = layer.identity
                probs[name] = layer.p
        return probs, identity

    @classmethod
    def set_stratified_dropout_identity(cls, model: torch.nn.Module, identity, strict: bool = False):
        """
        Assign each stratified layer its mask from ``identity``.

        With ``strict=True`` (used by :func:`apply_population`), every active
        stratified layer must have an entry in ``identity``; otherwise an error
        is raised rather than letting the layer lazily fabricate a fresh random
        mask. This matters for multi-forward generation, where a silently
        fabricated mask would persist for the entire decode.
        """
        for name, layer in model.named_modules():
            if isinstance(layer, StratifiedDropoutMC):
                if name in identity:
                    layer.identity = identity[name]
                elif strict:
                    raise ValueError(
                        f"No identity provided for stratified layer '{name}'. "
                        "The population does not match this model (see Population.validate)."
                    )

    @classmethod
    def has_mc_dropout(cls, model: torch.nn.Module) -> bool:
        return any(isinstance(m, DropoutMC) for m in model.modules())

    @classmethod
    def convert_dropouts(cls, model, stratified=True, p: float = 0.1, verbose=False):
        """
        Back-compat entry point. Converts native dropouts to the MC variant and,
        if none exist, inserts them. Prefer :func:`prepare_model` for new code.
        """
        dropout_cls = StratifiedDropoutMC if stratified else DropoutMC
        ctor = (lambda p, activate, _cls=dropout_cls, _pp=p: _cls(p=_pp, activate=False))
        replacement_dict = {"Dropout": ctor}

        replaced_layers = cls._convert_to_mc_dropout(model, replacement_dict)

        if replaced_layers == 0:
            if verbose:
                print('trying to add dropout layers...')
            cls.add_new_dropout_layers(model, dropout_cls=dropout_cls, p=p, verbose=verbose)
            replaced_layers = cls._convert_to_mc_dropout(model, replacement_dict)
            # Inserted layers may already be MC dropouts (not native "Dropout"),
            # so also count those.
            if replaced_layers == 0 and cls.has_mc_dropout(model):
                replaced_layers = sum(
                    1 for m in model.modules() if isinstance(m, DropoutMC)
                )

        if verbose:
            cls.show_model(model)
            print('replaced ', replaced_layers, ' layers.')

        if replaced_layers == 0:
            raise ValueError(
                "The number of converted layers is zero. This is because the model "
                "has no dropout layers. Add them using add_new_dropout_layers()."
            )


# ---------------------------------------------------------------------------
# Convenience layer: prepare / activation control
# ---------------------------------------------------------------------------
@dataclass
class PrepareReport:
    already_prepared: bool
    converted: int
    inserted: int
    activated: bool
    layer_names: List[str] = field(default_factory=list)


def prepare_model(
    model: torch.nn.Module,
    *,
    where: Optional[Callable[[str, torch.nn.Module], bool]] = None,
    dropout_cls: type = StratifiedDropoutMC,
    p: float = 0.1,
    activate: bool = True,
    hidden_size: Optional[int] = None,
    verbose: bool = False,
) -> PrepareReport:
    """
    Idempotently prepare ``model`` for population sampling.

    Steps (skipped wholesale if the model already contains MC dropout layers):
      1. Convert native ``Dropout`` layers to ``dropout_cls``.
      2. If none were found, insert dropout structurally via ``where``.
    Then, if ``activate``, switch dropout on at probability ``p``.

    Safe to call repeatedly (no double-wrapping). This is the single point where
    insertion, conversion, and activation converge; ``generate_population`` and
    ``apply_population`` call it internally.
    """
    already = DropoutUtils.has_mc_dropout(model)
    converted = 0
    inserted = 0

    if not already:
        ctor = (lambda p, activate, _cls=dropout_cls, _pp=p: _cls(p=_pp, activate=False))
        converted = DropoutUtils._convert_to_mc_dropout(model, {"Dropout": ctor})

        if converted == 0:
            inserted = DropoutUtils.add_new_dropout_layers(
                model, where=where, dropout_cls=dropout_cls, p=p,
                hidden_size=hidden_size, verbose=verbose,
            )
            # Newly inserted layers are already MC dropouts; no second pass needed.
            if (converted + inserted) == 0:
                raise ValueError(
                    "Could not prepare the model: no dropout layers were found or "
                    "inserted. Provide a custom `where` predicate identifying the "
                    "feed-forward output projections for this architecture."
                )

    if activate:
        DropoutUtils.activate_mc_dropout(model, activate=True, random=p)

    layer_names = [n for n, m in model.named_modules() if isinstance(m, StratifiedDropoutMC)]
    return PrepareReport(
        already_prepared=already, converted=converted, inserted=inserted,
        activated=bool(activate), layer_names=layer_names,
    )


def deactivate(model: torch.nn.Module):
    """Switch all MC dropouts off (masks bypassed) for a deterministic baseline pass."""
    DropoutUtils.activate_mc_dropout(model, activate=False)


@contextmanager
def deactivated(model: torch.nn.Module):
    """Temporarily switch MC dropout off, restoring prior per-layer state on exit."""
    states = [(m, m.activate) for m in model.modules() if isinstance(m, DropoutMC)]
    for m, _ in states:
        m.activate = False
    try:
        yield model
    finally:
        for m, st in states:
            m.activate = st


# ---------------------------------------------------------------------------
# Population: a first-class, serialisable population object
# ---------------------------------------------------------------------------
class Population:
    """
    A reusable, savable population of individuals.

    Attributes
    ----------
    identities : list of dict[str, Tensor]
        One entry per member; each maps a stratified layer name to its
        feature-only mask (shape ``[1, ..., 1, H]``).
    p : dict[str, float]
        Per-layer dropout probability recorded at generation time. (Mask scaling
        is frozen at generation time, so at apply time ``p`` need only be
        non-zero; restoring the recorded value keeps members consistent.)
    fingerprint : dict
        ``{model_class, dropout_cls, layer_dims: {name: H}}`` used by
        :meth:`validate` to guard against applying a population to the wrong
        model (which would otherwise silently produce a degenerate result).
    """

    FORMAT_VERSION = 1

    def __init__(self, identities: List[Dict[str, torch.Tensor]],
                 p: Dict[str, float], fingerprint: Dict[str, Any]):
        self.identities = identities
        self.p = p
        self.fingerprint = fingerprint

    def __len__(self):
        return len(self.identities)

    def __iter__(self):
        return iter(self.identities)

    def __getitem__(self, i):
        return self.identities[i]

    def save(self, path: str):
        payload = {
            "format_version": self.FORMAT_VERSION,
            "identities": [
                {k: v.detach().to("cpu") for k, v in ident.items()}
                for ident in self.identities
            ],
            "p": dict(self.p),
            "fingerprint": self.fingerprint,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "Population":
        payload = torch.load(path, map_location=map_location)
        return cls(
            identities=payload["identities"],
            p=payload["p"],
            fingerprint=payload["fingerprint"],
        )

    def to(self, device) -> "Population":
        self.identities = [
            {k: v.to(device) for k, v in ident.items()} for ident in self.identities
        ]
        return self

    def bind(self, model: torch.nn.Module, **kwargs) -> "PopulationModel":
        """
        Bind this population to a live (prepared) ``model`` and return a
        :class:`PopulationModel` whose ``forward`` / ``generate`` / ``__call__``
        emit population-shaped outputs. Convenience for
        ``PopulationModel(model, population, **kwargs)``.
        """
        return PopulationModel(model, self, **kwargs)

    def validate(self, model: torch.nn.Module, strict_class: bool = True):
        """
        Check that this population matches ``model``: every recorded stratified
        layer is present on the model and vice versa, and (optionally) the model
        class matches. Raises ``ValueError`` on mismatch.

        Feature-dimension (H) mismatches that slip past the name check surface at
        run time when the mask is broadcast against the activation, so they
        cannot silently corrupt results.
        """
        model_layers = {
            n for n, m in model.named_modules() if isinstance(m, StratifiedDropoutMC)
        }
        pop_layers = set(self.fingerprint.get("layer_dims", {}).keys())

        missing = pop_layers - model_layers
        if missing:
            raise ValueError(
                f"Population references {len(missing)} stratified layer(s) absent "
                f"from the model, e.g. {sorted(missing)[:3]}. The model is not "
                "prepared the same way the population was generated."
            )
        extra = model_layers - pop_layers
        if extra:
            raise ValueError(
                f"Model has {len(extra)} stratified layer(s) not covered by the "
                f"population, e.g. {sorted(extra)[:3]}. Applying it would leave "
                "those layers to fabricate uncontrolled random masks."
            )

        if strict_class:
            expected = self.fingerprint.get("model_class")
            actual = type(model).__name__
            if expected is not None and expected != actual:
                raise ValueError(
                    f"Population was generated on '{expected}' but is being applied "
                    f"to '{actual}'. Pass strict_class=False to override."
                )


def _make_fingerprint(model, identity, dropout_cls) -> Dict[str, Any]:
    cls_name = dropout_cls.__name__ if isinstance(dropout_cls, type) else type(dropout_cls).__name__
    return {
        "model_class": type(model).__name__,
        "dropout_cls": cls_name,
        "layer_dims": {name: int(mask.shape[-1]) for name, mask in identity.items()},
    }


def _restore_population_state(model, population: Population):
    """Set activate=True and restore each layer's recorded p, so masks bite."""
    for name, mod in model.named_modules():
        if isinstance(mod, StratifiedDropoutMC) and name in population.p:
            recorded = population.p[name]
            # Mask scaling is frozen at generation time; p need only be non-zero.
            mod.p = recorded if recorded else 0.1
            mod.p_init = mod.p
            mod.activate = True


def _align_population_device(model, population: Population):
    """
    Move a population's masks onto the model's (primary) parameter device.

    A population saved/loaded via torch.save lands on CPU; applying it to a CUDA
    model would otherwise raise on the mask multiply. This assumes a single-device
    model -- for ``device_map='auto'`` sharded across several GPUs, load the
    population with a matching ``map_location`` or split it per layer-device.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return
    population.to(device)


# ---------------------------------------------------------------------------
# Convenience layer: generate / apply
# ---------------------------------------------------------------------------
def generate_population(
    model: torch.nn.Module,
    calibrate: Callable[[], Any],
    committee_size: int = 20,
    *,
    p: float = 0.1,
    where: Optional[Callable[[str, torch.nn.Module], bool]] = None,
    dropout_cls: type = StratifiedDropoutMC,
    hidden_size: Optional[int] = None,
    prepare: bool = True,
) -> Population:
    """
    Create a reusable, batch-independent :class:`Population`.

    ``calibrate`` is a zero-argument callable that runs a single forward pass; it
    exists only so each layer can learn its feature width ``H``. Because masks are
    feature-only, the *content* of the calibration input does not affect the
    resulting population, and the population can later be applied to data of any
    batch size or sequence length.
    """
    if prepare:
        prepare_model(model, where=where, dropout_cls=dropout_cls, p=p,
                      activate=True, hidden_size=hidden_size)

    DropoutUtils.reset_stratified_mc_dropout(model)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            calibrate()
    finally:
        if was_training:
            model.train()

    probs, initial_identity = DropoutUtils.get_stratified_dropout_identity(model)
    if not initial_identity:
        raise ValueError(
            "No stratified dropout layers were instantiated. Ensure the model is "
            "prepared (prepare=True) and that `calibrate` actually runs a forward."
        )

    # Member 0 is the mask produced by the calibration forward; clone so later
    # member assignments cannot mutate it.
    identities: List[Dict[str, torch.Tensor]] = [
        {name: mask.clone() for name, mask in initial_identity.items()}
    ]

    for _ in range(committee_size - 1):
        new_identity = {}
        for name, mask in initial_identity.items():
            pp = probs[name]
            new_identity[name] = mask.data.new(torch.Size(mask.shape)).bernoulli_(1 - pp).div_(1 - pp)
        identities.append(new_identity)

    fingerprint = _make_fingerprint(model, initial_identity, dropout_cls)
    return Population(identities=identities, p=probs, fingerprint=fingerprint)


def apply_population(
    model: torch.nn.Module,
    population: Population,
    fn: Callable[[], Any],
    *,
    validate: bool = True,
    strict_class: bool = True,
    enforce_eval: bool = True,
    seed: Optional[int] = None,
    samples_per_member: int = 1,
):
    """
    Run ``fn`` once per member (optionally several times) and yield each result.

    ``fn`` is responsible for iterating the *entire* dataset internally (outer
    loop = member here, inner loop = batches in ``fn``). For free generation,
    ``fn`` performs the full decode per stimulus; nothing in this loop resets a
    member's mask mid-call, so each member is one consistent reasoner across its
    whole trace.

    Generation-related guarantees
    -----------------------------
    * The population is validated against the model before any forward.
    * Dropout is activated and each layer's recorded ``p`` restored, so a loaded
      population can never silently run against an inert model.
    * ``enforce_eval`` puts the model in eval mode for the duration, so the
      stratified masks are the sole source of randomness (any unconverted native
      dropouts become no-ops); the prior train/eval state is restored on exit.
    * ``seed``/``samples_per_member`` control decoding stochasticity. With
      deterministic decoding each member is one reproducible subject. With
      sampled decoding, set ``seed`` (common random numbers are reused across
      members for a given sample index, isolating mask-driven variance) and use
      ``samples_per_member`` to draw an explicit, labelled second axis of
      within-member samples.
    """
    if validate:
        population.validate(model, strict_class=strict_class)

    _align_population_device(model, population)
    _restore_population_state(model, population)

    prev_mode_training = model.training
    if enforce_eval:
        model.eval()
    try:
        for identity in population.identities:
            DropoutUtils.set_stratified_dropout_identity(model, identity, strict=True)
            for s in range(samples_per_member):
                if seed is not None:
                    torch.manual_seed(seed + s)
                yield fn()
    finally:
        if enforce_eval and prev_mode_training:
            model.train()


# ---------------------------------------------------------------------------
# Legacy primitives (kept for back-compat; now produce feature-only masks)
# ---------------------------------------------------------------------------
def generate_dropout_population(model, call_to_model_lambda, committee_size=20):
    identities = []
    DropoutUtils.reset_stratified_mc_dropout(model)
    call_to_model_lambda()

    probs, initial_identity = DropoutUtils.get_stratified_dropout_identity(model)
    identities.append(initial_identity)

    for index in range(committee_size - 1):
        new_identity = {}

        for layer in initial_identity.keys():
            p = probs[layer]

            tens = initial_identity[layer]
            new = tens.data.new(torch.Size(tens.shape)).bernoulli_(1 - p).div_(1 - p)
            new_identity[layer] = new

        identities.append(new_identity)

    return identities


def call_function_with_population(model, identities, function_to_call):
    for identity in identities:
        DropoutUtils.set_stratified_dropout_identity(model, identity)
        yield function_to_call()


def generate_population_and_apply(model, function_to_call, committee_size=20, transpose=False):
    outs = list(call_function_with_population(
        model, generate_dropout_population(model, function_to_call, committee_size), function_to_call))
    if transpose and isinstance(outs[0], list):
        outs = list(map(list, zip(*outs)))
    return outs

# ---------------------------------------------------------------------------
# Output stacking: turn a list of per-member outputs into a single
# population-shaped structure (new leading axis P by default).
# ---------------------------------------------------------------------------
def _is_model_output(obj) -> bool:
    """Duck-type a HF ``ModelOutput`` (an OrderedDict subclass) without importing
    transformers: it is dict-like and exposes ``to_tuple``."""
    return isinstance(obj, dict) and hasattr(obj, "to_tuple")


def _stack_tensors(items: List[torch.Tensor], dim: int, pad_value) -> torch.Tensor:
    """Stack tensors on a new axis ``dim``. If they disagree in shape (e.g. ragged
    generation lengths), right-pad each to the elementwise-max shape first."""
    shapes = [tuple(t.shape) for t in items]
    if all(s == shapes[0] for s in shapes):
        return torch.stack(items, dim=dim)

    ndim = items[0].dim()
    if any(t.dim() != ndim for t in items):
        # Genuinely incompatible ranks -- refuse to guess; hand back a list.
        return list(items)  # type: ignore[return-value]
    max_shape = [max(s[d] for s in shapes) for d in range(ndim)]
    padded = []
    for t in items:
        pad = []
        for d in reversed(range(ndim)):          # F.pad consumes last dim first
            pad.extend([0, max_shape[d] - t.shape[d]])
        padded.append(torch.nn.functional.pad(t, pad, value=pad_value))
    return torch.stack(padded, dim=dim)


def stack_outputs(items, dim: int = 0, *, pad_value=0,
                  drop_keys=("past_key_values",)):
    """
    Recursively combine a list of per-member outputs into one structure with a
    new population axis at ``dim``.

    * tensors          -> stacked (ragged tensors are right-padded with
                          ``pad_value``; the common case is ``generate`` output of
                          differing lengths).
    * ModelOutput/dict -> same type rebuilt field-wise; ``drop_keys`` (default
                          ``past_key_values``, which is per-member and not
                          meaningfully stackable) are omitted.
    * list/tuple       -> same type, stacked element-wise (e.g. ``hidden_states``).
    * anything else / mixed / None / ragged-rank -> returned as a plain list along
                          the member axis (never raises).
    """
    items = list(items)
    if len(items) == 0:
        return items
    first = items[0]

    if isinstance(first, torch.Tensor) and all(isinstance(it, torch.Tensor) for it in items):
        return _stack_tensors(items, dim, pad_value)

    if _is_model_output(first):
        fields = {}
        for k in first.keys():
            if k in drop_keys:
                continue
            vals = [it.get(k) for it in items]
            if all(v is None for v in vals):
                continue
            if any(v is None for v in vals):
                fields[k] = vals                 # mixed presence -> keep as list
            else:
                fields[k] = stack_outputs(vals, dim=dim, pad_value=pad_value,
                                          drop_keys=drop_keys)
        return type(first)(**fields)

    if isinstance(first, dict):
        keys = first.keys()
        return {k: stack_outputs([it[k] for it in items], dim=dim,
                                 pad_value=pad_value, drop_keys=drop_keys) for k in keys}

    if isinstance(first, (list, tuple)):
        if any(not isinstance(it, (list, tuple)) or len(it) != len(first) for it in items):
            return items                         # ragged structure -> list
        stacked = [stack_outputs([it[i] for it in items], dim=dim,
                                 pad_value=pad_value, drop_keys=drop_keys)
                   for i in range(len(first))]
        return tuple(stacked) if isinstance(first, tuple) else stacked

    # scalars, strings, None, or anything we can't stack.
    return items


def _reshape_leading(obj, groups: int, batch: int, drop_keys=("past_key_values",)):
    """Map a structure whose tensors have leading dim ``groups*batch`` (member-major)
    back to leading ``[groups, batch, ...]``. Used to post-process a single
    vectorised forward into the same shape the per-member loop produces."""
    if isinstance(obj, torch.Tensor):
        if obj.dim() >= 1 and obj.shape[0] == groups * batch:
            return obj.view(groups, batch, *obj.shape[1:])
        return obj
    if _is_model_output(obj):
        fields = {}
        for k in obj.keys():
            if k in drop_keys or obj.get(k) is None:
                continue
            fields[k] = _reshape_leading(obj[k], groups, batch, drop_keys)
        return type(obj)(**fields)
    if isinstance(obj, dict):
        return {k: _reshape_leading(v, groups, batch, drop_keys) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        z = [_reshape_leading(v, groups, batch, drop_keys) for v in obj]
        return tuple(z) if isinstance(obj, tuple) else z
    return obj


# ---------------------------------------------------------------------------
# PopulationModel: a transformers-like facade over a base model + Population.
# ---------------------------------------------------------------------------
class PopulationModel:
    """
    Bind a prepared base model to a :class:`Population` and expose a
    transformers-like surface. The "run the model" entry points
    (``__call__`` / ``forward`` and ``generate``) are *population-aware*: they run
    the underlying call once per member and stack the results on a new leading
    population axis (configurable via ``population_dim``). Every other attribute
    (``config``, ``device``, ``get_input_embeddings``, ``to``, ``eval`` ...) falls
    through to the wrapped model unchanged.

    Examples
    --------
    >>> pm = population.bind(model)                     # or PopulationModel(model, population)
    >>> out = pm(**inputs)                              # out.logits: [P, B, T, V]
    >>> mean_logits = out.logits.mean(0)               # distributional summary
    >>> gen = pm.generate(**inputs, max_new_tokens=32) # [P, B, T] (padded)

    Per-member post-processing & memory
    -----------------------------------
    Materialising ``[P, B, T, vocab]`` logits is often wasteful. Pass ``reduce=fn``
    to apply ``fn`` to each member's raw output *before* stacking, so only the
    reduced quantity (e.g. continuation log-probs, shape ``[P, B]``) is kept:

    >>> scores = pm(input_ids, reduce=lambda o: o.logits.log_softmax(-1)[..., -1, :])

    Pass ``stack=False`` to get a plain list of per-member outputs (lazy-friendly,
    full control). ``seed`` reuses common random numbers across members so that,
    under sampled decoding, mask-driven variance is isolated from decoding noise.

    Notes
    -----
    * By default this is ``P`` sequential forwards behind a single call. Pass
      ``vectorized=True`` to ``forward``/``__call__`` to run a single tiled pass
      instead (one kernel launch, ``P x`` activation memory; forward only). It is
      exact here because the masks sit at FFN outputs and every decoder op is
      independent across the batch axis.
    * ``enforce_eval`` (default) holds the model in eval mode for each call so the
      stratified masks are the only randomness; prior mode is restored after.
    """

    # Attribute lookups that should NOT fall through to the wrapped model.
    _OWN_ATTRS = frozenset({"model", "population", "population_dim", "enforce_eval"})

    def __init__(self, model: torch.nn.Module, population: Population, *,
                 population_dim: int = 0, validate: bool = True,
                 strict_class: bool = True, enforce_eval: bool = True):
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "population", population)
        object.__setattr__(self, "population_dim", population_dim)
        object.__setattr__(self, "enforce_eval", enforce_eval)
        if validate:
            population.validate(model, strict_class=strict_class)
        _align_population_device(model, population)
        _restore_population_state(model, population)

    # -- introspection -----------------------------------------------------
    def __len__(self):
        return len(self.population)

    def __repr__(self):
        return (f"PopulationModel(members={len(self.population)}, "
                f"model={type(self.model).__name__}, population_dim={self.population_dim})")

    def __getattr__(self, name):
        # Only reached when normal lookup fails -> delegate to the base model.
        # (self.model is set via object.__setattr__ in __init__, so this is safe.)
        return getattr(object.__getattribute__(self, "model"), name)

    # -- core member loop --------------------------------------------------
    def _run(self, call, *, reduce=None, stack=True, seed=None,
             population_dim=None, pad_value=0):
        prev_training = self.model.training
        if self.enforce_eval:
            self.model.eval()
        outs = []
        try:
            for identity in self.population.identities:
                DropoutUtils.set_stratified_dropout_identity(self.model, identity, strict=True)
                if seed is not None:
                    torch.manual_seed(seed)
                with torch.no_grad():
                    o = call()
                outs.append(reduce(o) if reduce is not None else o)
        finally:
            if self.enforce_eval and prev_training:
                self.model.train()
        if not stack:
            return outs
        dim = self.population_dim if population_dim is None else population_dim
        return stack_outputs(outs, dim=dim, pad_value=pad_value)

    def map(self, fn, *, stack=True, seed=None, population_dim=None):
        """Run an arbitrary ``fn()`` (no args) once per member -- the general
        escape hatch, equivalent to :func:`apply_population` but bound."""
        return self._run(fn, stack=stack, seed=seed, population_dim=population_dim)

    # -- population-aware entry points -------------------------------------
    def forward(self, *args, reduce=None, stack=True, seed=None,
                vectorized=False, population_dim=None, **kwargs):
        if vectorized:
            return self._forward_vectorized(args, kwargs, reduce=reduce,
                                            population_dim=population_dim)
        return self._run(lambda: self.model(*args, **kwargs),
                         reduce=reduce, stack=stack, seed=seed,
                         population_dim=population_dim)

    __call__ = forward

    def generate(self, *args, reduce=None, stack=True, seed=None,
                 population_dim=None, pad_value=None, **kwargs):
        """Population ``generate``. Returns stacked sequences (a ``LongTensor``
        of shape ``[P, B, T]``, right-padded across members with ``pad_value``)
        or, when ``return_dict_in_generate=True``, a stacked ``GenerateOutput``
        with the same population axis (ragged per-step fields fall back to lists).
        ``pad_value`` defaults to the model's ``pad_token_id`` (then ``eos_token_id``)."""
        if pad_value is None:
            cfg = getattr(self.model, "config", None)
            pad_value = getattr(cfg, "pad_token_id", None) if cfg is not None else None
            if pad_value is None:
                pad_value = getattr(cfg, "eos_token_id", 0) if cfg is not None else 0
            if isinstance(pad_value, (list, tuple)):
                pad_value = pad_value[0]
        return self._run(lambda: self.model.generate(*args, **kwargs),
                         reduce=reduce, stack=stack, seed=seed,
                         population_dim=population_dim, pad_value=pad_value)

    # -- optional single-pass (vectorised) forward -------------------------
    def _forward_vectorized(self, args, kwargs, *, reduce=None, population_dim=None):
        """One tiled forward instead of P sequential ones. Tiles the batch by P
        (member-major), installs the whole population's masks as stacked buffers,
        runs once, then reshapes the leading P*B axis back to [P, B, ...]."""
        P = len(self.population)
        batch = _infer_vectorized_batch(args, kwargs)
        if batch is None:
            raise ValueError("vectorized=True needs a batched tensor input "
                             "(input_ids / inputs_embeds) to infer the batch size.")

        tiled_args = tuple(_tile_batch(a, P, batch) for a in args)
        tiled_kwargs = {k: _tile_batch(v, P, batch) for k, v in kwargs.items()}

        layers = {n: m for n, m in self.model.named_modules()
                  if isinstance(m, StratifiedDropoutMC)}
        device = next(self.model.parameters()).device

        prev_training = self.model.training
        if self.enforce_eval:
            self.model.eval()
        try:
            for name, layer in layers.items():
                masks = [self.population.identities[i][name] for i in range(P)]
                layer._vector_identity = torch.stack(masks, dim=0).to(device)
                layer._vector_groups = P
                layer.activate = True
            with torch.no_grad():
                raw = self.model(*tiled_args, **tiled_kwargs)
        finally:
            for layer in layers.values():
                if hasattr(layer, "_vector_identity"):
                    del layer._vector_identity
                if hasattr(layer, "_vector_groups"):
                    del layer._vector_groups
            if self.enforce_eval and prev_training:
                self.model.train()

        out = _reshape_leading(raw, P, batch)
        if reduce is not None:
            out = reduce(out)
        if population_dim not in (None, 0):
            # Caller wants the population axis somewhere other than leading.
            out = stack_outputs([out_i for out_i in _split_leading(out, P)],
                                dim=population_dim)
        return out


def _infer_vectorized_batch(args, kwargs) -> Optional[int]:
    cand = kwargs.get("input_ids")
    if cand is None:
        cand = kwargs.get("inputs_embeds")
    if cand is None and args:
        cand = args[0]
    return cand.shape[0] if isinstance(cand, torch.Tensor) and cand.dim() >= 1 else None


def _tile_batch(v, groups: int, batch: int):
    """Repeat a batch-leading tensor ``groups`` times in member-major order."""
    if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == batch:
        return v.repeat(groups, *([1] * (v.dim() - 1)))
    return v


def _split_leading(obj, groups: int):
    """Yield ``groups`` slices along a leading population axis (for re-placing it)."""
    for i in range(groups):
        if isinstance(obj, torch.Tensor):
            yield obj[i]
        elif _is_model_output(obj) or isinstance(obj, dict):
            yield type(obj)(**{k: (v[i] if isinstance(v, torch.Tensor) else v)
                               for k, v in obj.items()}) if _is_model_output(obj) \
                  else {k: (v[i] if isinstance(v, torch.Tensor) else v) for k, v in obj.items()}
        else:
            yield obj


__all__ = [
    # primitives
    "DropoutMC", "StratifiedDropoutMC", "LockedDropoutMC", "WordDropoutMC",
    "MC_DROPOUT_SUBSTITUTES", "DropoutUtils",
    "generate_dropout_population", "call_function_with_population",
    "generate_population_and_apply",
    # insertion
    "default_insertion_predicate",
    # convenience layer
    "Population", "PopulationModel", "prepare_model", "deactivate", "deactivated",
    "generate_population", "apply_population", "stack_outputs",
]