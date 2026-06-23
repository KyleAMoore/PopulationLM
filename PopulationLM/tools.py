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

        identity_expanded = self.identity.expand_as(x)

        if not self.activate or not self.p:
            return x

        return identity_expanded * x


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