# Fixed nine-arm control-answer follow-up

This run fills the missing free-answer observations for the previously tested
controls. It replays one existing input and all six specified controls, including
controls that did not produce the same first-token change. No new prompt, prefix,
neuron selection, control search, training, gradients, or model change is allowed.

## Frozen source and input

Use the unchanged original request **r013**, including its exact prompt tokens,
chat template, request ID, and seed `20261117 + 13`. No instruction, answer cue,
or common response prefix is added. Generate from prompt IDs only.

Keep `Qwen/Qwen3-1.7B` revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, FP32, TF32 disabled,
SDPA/EFFICIENT_ATTENTION, evaluation mode, thinking off, and frozen parameters.
Keep the ordered H286, fixed REF-W statistics, and the local nine-coordinate G7.
The existing H286 selection was post hoc; G7 is its complete block-7 subset,
without additional outcome-based selection. Private coordinate IDs and prior
answer texts must remain in the private reference package, not this document.

The prior group-run archive has SHA256
`462b445b64d950fe525e70237d8f8636ca3db35e3d9bddee19d276aa331f85ad`;
its canonical plan SHA256 is
`c9bf357e934fe308363e6131415b316158e2ba542ab0f127adb701220a3ed10d`.
Read those sources and their original modules without modifying them. Bind a new
plan and new module hashes for this follow-up. Supply the four existing private
references (`hn_set.json`, `htq_policy.json`, `refw_h286.json`, and
`group_continuity.json`) plus a separately hashed private extract of the prior
group run. Validate their identities before model execution.

## Nine fixed arms and 21 prefill calls

| Arm | Operation |
|---|---|
| `sham` | Original clean/sham baseline |
| `other_block_108` | Original block-7 attention pulse, T=1.08 |
| `b7_transfer` | Clean recipient receives G7 values from the pulsed donor |
| `b7_outside_transfer_0`, `_1`, `_2` | All three original outside-G7 transfers |
| `b7_within_transfer_0`, `_1`, `_2` | All three original within-G7 random directions |

Use two fresh, same-prompt baseline donors: clean and block-7 pulse. The target
transfer assigns their signed gated-MLP z values directly at the original final
prompt position, immediately before `down_proj`. Never patch CETT or zCETT.
Outside controls retain the original masks and donor-difference directions;
within controls retain the original random directions. Preserve all source
seeds, signs, raw-norm scaling rules, and numerical guards. Recompute magnitudes
from the fresh donors by that same rule; never retune after seeing outputs.

Execute nine measured prefills, one genuinely uninstrumented clean prefill, and
two G7 identity checks (clean-to-clean and pulse-to-pulse). Then run nine fresh
generation prefills: **21 total prefill forwards**, followed by the selected
cached decode steps. The two donors are included in the nine measured arms.
This is a nine-arm follow-up, not a repetition or enlargement of the prior
35-arm design. No block-14 arm or additional control is part of this run.

Each answer uses a fresh cache and greedy decoding, capped at **128 selected
tokens including EOS**. Record selected EOS but exclude it from emitted text and
token IDs. Never force continuation after EOS. Pulse and patch operate only in
prefill; remove their hooks before cached decoding. No waiting GPU or queued
follow-up stage is needed: run checks, all nine arms, export, and finish.

## Compatibility and saved evidence

- Require all nine measured first-token IDs to match the corresponding prior
  group-run arms. Require the three previously generated arms to reproduce their
  recorded selected-token sequence and EOS/cap convention. A mismatch is an
  explicit compatibility failure; do not relabel it as a successful continuation.
- Record numerical continuity separately: old/current full-logit hashes, saved
  state arrays, query and patch norms, top-token probabilities, and available
  step-logprob comparisons. Numerical diagnostics are not interchangeable with
  the hard token/protocol conditions.
- Require each generation replay's first logits to match its current measured
  prefill. Check both identity arms, native clean equivalence, finite values,
  correct patch positions and channels, hook cleanup, and frozen parameters.
- Save **all nine full-vocabulary first-step FP32 logit vectors** in ordinary
  numeric NPZ arrays. At this vocabulary size they occupy about 5.5 MB before
  compression and permit independent offline KL recomputation. Do not store
  object arrays, pretrained weights, or full-vocabulary logits for every decode.
- Calculate `KL(reference || arm)` with stable FP64 log-softmax over the full
  vocabulary, using clean and pulse references for the identical prompt. State
  direction explicitly. Different generated histories are not identical-prefix
  distributions and must not be treated as such in a trajectory KL comparison.
- Preserve raw signed-z and fixed-REF readouts, exact absolute capture positions,
  patch before/after values, control masks/seeds, raw-norm errors, and projected
  `W_down * delta` norms/cosines. Matching raw norms does not imply matched output
  norms or equal functional damage. An undefined cosine is missing, not zero.
- Save every answer's exact text, selected/emitted tokens, EOS status, cap status,
  and per-step top tokens. Prompt captures remain prompt captures; this protocol
  does not add decoder activation trajectories or energy measurements.

## Analysis boundaries

Evaluate all nine answers, with a separate complete table for the outside-group
controls and for the within-group direction controls. Outside controls address
group membership; within controls address the specific transferred direction.
Do not pool them into a single success-rate statistic or select only controls
that crossed a first-token boundary. The three seeds are not independent tasks.

Keep semantic endorsement, strict surface formatting, EOS completion, and
truncation separate. A token mentioned in a quoted premise or draft is not an
endorsed answer. A noncanonical explanation may be semantically correct; a capped
answer remains incomplete even if its current text contains a correct statement.
Any manual semantic interpretation should quote the relevant conclusion and
retain ambiguous or contradictory cases explicitly.

The question is whether the fixed control transfers also produce the relevant
complete-answer behavior, rather than merely a similar opening token. The result
is conditional evidence about this one previously inspected prompt. It is not a
held-out test, a generalization estimate, a community-discovery result, the full
community necessity/sufficiency battery, or evidence of self-monitoring or memory.
Report actual wall time; make no energy or fixed-completion-time claim.
