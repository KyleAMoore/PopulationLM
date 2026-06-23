"""
PopulationLM -- usage example for the PopulationModel (transformers-like) API.

Demonstrates the common patterns:
  1. population creation (once)
  2. save / load / validate
  3. single-token / cloze inference            -> pm(...) with a `reduce` hook
  4. a batched population forward               -> pm(**inputs).logits is [P, B, T, V]
  5. autoregressive generation                  -> pm.generate(...)  is [P, B, T]
  6. the retained primitive                     -> apply_population(model, pop, fn)

NOTE: `import PopulationLM as pop` assumes PopulationLM/__init__.py re-exports the
public API, e.g. `from .tools import *` (tools.py now defines __all__). If you are
running against tools.py directly, use `import tools as pop` instead.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import PopulationLM as pop

hf_token = None  # omitted
MODEL = "meta-llama/Llama-3.2-1B"
COMMITTEE_SIZE = 5
DROPOUT_P = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"
 
config = AutoConfig.from_pretrained(MODEL, token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(MODEL, clean_up_tokenization_spaces=False)
# clean_up_tokenization_spaces is a WordPiece post-process; it is destructive for
# BPE (Llama), stripping spaces before punctuation -> set False for faithful decode.
# LLaMA tokenizers ship without a pad token; batched tokenization/generation need one.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Decoder-only models should left-pad so the true final token stays at index -1
# (right padding would put a pad token there for shorter sequences in a batch).
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    MODEL, config=config, token=hf_token, low_cpu_mem_usage=True, device_map="auto"
)
model = model.to(device)
model.eval()
 
 
# ---------------------------------------------------------------------------
# 1. Generate the population ONCE.
#    LLaMA ships without usable dropout, so generate_population (via prepare_model)
#    inserts stratified dropout at each FFN down-projection automatically. The
#    calibration forward only learns each layer's feature width -- its input
#    content does not affect the population (masks are batch/sequence independent).
# ---------------------------------------------------------------------------
calibrate = lambda: model(**tokenizer("calibration text", return_tensors="pt").to(device))
 
population = pop.generate_population(
    model, calibrate, committee_size=COMMITTEE_SIZE, p=DROPOUT_P
)
print(f"Generated a population of {len(population)} individuals.")
 
 
# ---------------------------------------------------------------------------
# 2. Save and load the population (portable + batch-independent).
# ---------------------------------------------------------------------------
population.save("/content/drive/MyDrive/Research/population.pt")
population = pop.Population.load("/content/drive/MyDrive/Research/population.pt")  # round-trip
population.validate(model)                         # guard vs. a mismatched model
 
 
# ---------------------------------------------------------------------------
# Bind the population to the model -> a transformers-like PopulationModel.
#   * pm(...) / pm.forward(...) and pm.generate(...) are population-aware: they run
#     once per member and stack results on a new leading population axis (P).
#   * binding validates, aligns the population onto the model's device, and
#     activates the masks -- a loaded population can never silently run inert.
#   * every other attribute (pm.config, pm.device, pm.get_input_embeddings, ...)
#     falls through to the underlying model.
# ---------------------------------------------------------------------------
pm = population.bind(model)
print(pm)
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def score_continuation(prefix: str, continuation: str) -> torch.Tensor:
    """Per-member mean log-prob of `continuation` given `prefix` (cloze-style).
 
    One forward per member, but only the scalar score is kept per member (the
    `reduce` hook runs before stacking), so we never materialise [P, T, vocab]
    logits. Returns a tensor of shape [P].
    """
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(prefix + continuation, return_tensors="pt").input_ids.to(device)
    n_prefix = prefix_ids.shape[1]
    n_cont = full_ids.shape[1] - n_prefix
 
    targets = full_ids[0, n_prefix:]                                    # [n_cont]
    positions = torch.arange(n_prefix - 1, n_prefix - 1 + n_cont, device=device)
 
    def reduce(out):
        logprobs = out.logits.log_softmax(dim=-1)                      # [1, T, V]
        chosen = logprobs[0, positions, targets]                       # [n_cont]
        return chosen.mean()                                           # 0-dim tensor
 
    return pm(full_ids, reduce=reduce)                                 # [P]
 
 
def complete(prompt: str, max_new_tokens: int = 20) -> list[str]:
    """Greedy autoregressive completion for EVERY member at once.
 
    pm.generate runs the full decode per member with that member's mask held
    fixed across all steps, then stacks (right-padding ragged lengths). Returns
    one decoded string per member.
    """
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    gen = pm.generate(                                                 # [P, B, T] (padded)
        enc.input_ids,
        attention_mask=enc.attention_mask,  # pad_token == eos: mask can't be inferred, pass it
        max_new_tokens=max_new_tokens,
        do_sample=False,                  # deterministic: variation is purely the population
        pad_token_id=tokenizer.eos_token_id,
        pad_value=tokenizer.eos_token_id, # value used to pad ragged members when stacking
    )
    start = enc.input_ids.shape[1]
    return [tokenizer.decode(gen[m, 0, start:], skip_special_tokens=True)
            for m in range(len(pm))]
 
 
# ---------------------------------------------------------------------------
# 3A. SINGLE-INFERENCE cloze task.
#     Call the model directly; the population axis comes back automatically.
#     The natural payoff is a distribution per item (mean / std over individuals).
# ---------------------------------------------------------------------------
cloze_items = [
    ("The capital of France is", " Paris"),
    ("A robin is a", " bird"),
    ("Two plus two equals", " four"),
]
 
print("\n=== (A) Cloze: population log-prob per item ===")
for prefix, cont in cloze_items:
    scores = score_continuation(prefix, cont)                          # [P]
    per_member = [round(v, 3) for v in scores.tolist()]
    print(f"  {prefix!r} + {cont!r}:  mean={scores.mean():.3f}  "
          f"std={scores.std():.3f}  per-member={per_member}")
 
 
# ---------------------------------------------------------------------------
# 3B. A BATCHED population forward in a single expression.
#     pm(**inputs).logits has shape [P, B, T, V]; distributional summaries are
#     then just reductions over axis 0.
# ---------------------------------------------------------------------------
print("\n=== (B) Batched forward -> population logits ===")
batch = tokenizer(["The sky is", "Water is made of"],
                  return_tensors="pt", padding=True).to(device)
out = pm(**batch)                                                      # population-aware
print("  logits shape:", tuple(out.logits.shape))                     # [P, B, T, V]
next_token_logits = out.logits[:, :, -1, :]                           # [P, B, V]
mean_next = next_token_logits.mean(0)                                  # population mean: [B, V]
for b in range(mean_next.shape[0]):
    tok = tokenizer.decode(mean_next[b].argmax())
    print(f"    item {b}: population-mean next token -> {tok!r}")
 
# One-pass variant (forward only): a single tiled forward instead of P sequential
# ones. Identical result, one kernel launch, ~P x activation memory. Useful for
# small batches; for large vocab/batch keep the default loop (or add a `reduce`).
#   out_fast = pm(**batch, vectorized=True)        # same [P, B, T, V]
 
 
# ---------------------------------------------------------------------------
# 3C. AUTOREGRESSIVE completion -- SAME population object.
#     Each member completes every prompt under its own fixed mask.
# ---------------------------------------------------------------------------
prompts = ["Once upon a time,", "The best way to learn a language is"]
 
print("\n=== (C) Sentence completion (greedy), one block per individual ===")
for prompt in prompts:
    completions = complete(prompt)                                    # list of P strings
    print(f"  {prompt!r}")
    for m, text in enumerate(completions):
        print(f"    member {m}: {text!r}")
 
# For stochastic decoding, set do_sample=True in `complete` and pass seed=0 to
# pm.generate: common random numbers are reused across members so the variance you
# see is mask-driven, not decoding noise. Draw K samples per member by calling
# generate K times (or loop with different seeds).
 
 
# ---------------------------------------------------------------------------
# 4. The retained primitive: apply_population(model, population, fn).
#    Still useful when you want full manual control of the per-member loop, or to
#    reuse existing single-model code. Here `fn` uses the BASE model directly --
#    apply_population sets each member's mask before each call (do NOT nest pm
#    inside fn, or you would loop the population twice).
# ---------------------------------------------------------------------------
def score_continuation_base(prefix: str, continuation: str) -> float:
    """Mean log-prob under the base model's CURRENT mask (set by apply_population)."""
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(prefix + continuation, return_tensors="pt").input_ids.to(device)
    n_prefix = prefix_ids.shape[1]
    n_cont = full_ids.shape[1] - n_prefix
    with torch.no_grad():
        logprobs = model(full_ids).logits.log_softmax(dim=-1)
    total = 0.0
    for i in range(n_cont):
        pos = n_prefix + i - 1
        tok = full_ids[0, n_prefix + i]
        total += logprobs[0, pos, tok].item()
    return total / max(n_cont, 1)
 
print("\n=== (D) Legacy apply_population (manual per-member loop) ===")
score_all = lambda: [score_continuation_base(pre, cont) for pre, cont in cloze_items]
for i, scores in enumerate(pop.apply_population(model, population, score_all)):
    print(f"  member {i}: " + ", ".join(f"{round(s, 3)}" for s in scores))
