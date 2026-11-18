# Temporal propagation and a frozen functional intervention

This exploratory stage asks whether a brief attention intervention leaves a
measurable trace across an identical continuation, and whether one direction
estimated on calibration inputs changes later answer preferences on separate
test inputs. It follows the [local transport](B14_B18_LOCAL_TRANSPORT.md) and
[damping refinement](B14_B18_REFINEMENT.md) stages. It does not fit another
per-test target reconstruction, train weights or discover neuron communities.

## Frozen model, inputs and time axis

Use `Qwen/Qwen3-1.7B` revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, native FP32, TF32 disabled and SDPA
MATH throughout. Retain the original H286 coordinates and fixed REF-W statistics.
H286 was selected historically post hoc; this stage does not reselect it.
Model identity, sources, private inputs, split and checkpoints are hash-bound.

There are eight calibration inputs and eight test inputs. Each split contains
four knowledge questions (S3) and four deterministic arithmetic questions
(A-det). Historical outputs may already be known. Test means excluded from
this stage's candidate fitting, not globally unseen or independently sampled
from all previous project work. No direction, sign or amplitude is selected
using the new test outcomes.

Every input receives the same pretokenized, answer-free 32-token bridge,
beginning “I will consider...” and ending “Final answer:”. Its exact text and
token IDs are bound in the private input plan. This is a forced assistant
continuation, not a sampled reasoning trace or the original direct-answer
context. It changes the task context; conclusions must name that limitation.
It contains no clean/pulse label and is identical across all conditions.

All block and token indices are zero-based. If the original prompt has `L`
tokens, offset `k` means exactly `k` bridge tokens have been consumed:

| Readout offset | Absolute token position | Native cache length |
| --- | --- | --- |
| 0 | `L-1` | `L` |
| 1 | `L` | `L+1` |
| 8 | `L+7` | `L+8` |
| 32 | `L+31` | `L+32` |

After prompt prefill, use bridge chunks of 1, 7, 23 and 1 tokens. The final
one-token chunk permits a precisely timed endpoint intervention. Clean and
pulse use identical IDs and absolute positions. Pulse scales the post-`q_norm`
query in block14 at the original prompt position `L-1` once, with temperature
`T=1.08` and scale `beta=1/T`. Later queries are unpulsed. Full-prefix rebuilds
also pulse that original position, never the last bridge token.

## Readouts and the calibration-only candidate

At offsets0, 1, 8 and32, retain H286 signed raw MLP down-projection-input values,
CETT, fixed-reference zCETT, raw denominators and decoder residual outputs
13/14/15/17/18/19. These are different readouts: CETT includes absolute activation,
output-weight norms and a shared layer denominator. A zCETT change alone does
not establish signed activation propagation or a particular neuron's causal role.

For calibration input `i`, let `delta_i(k)` be pulse-minus-clean residual output
18 at offset `k`. Compute once from the eight calibration inputs:

```
u8 = mean_i(delta_i(8)) / norm(mean_i(delta_i(8)))
a32 = mean_i(dot(u8, delta_i(32)))
v32 = a32 * u8
```

Preserve the sign of `a32`; do not take its absolute value, flip its direction
or choose a favorable late amplitude. Freeze and hash the FP32 direction,
signed vector, source bindings and eight Gaussian control directions before
functional test evaluation. Each control has norm `norm(v32)` and is reused
across test inputs and the two intervention signs. It is not refitted per input.

Numerical informativeness gates compare the mean offset8 norm and absolute
late projection with `max(1e-6, 8 * largest calibration cache/rebuild L2 error)`
at the respective offset. A failed gate is reported: shared/noise interventions
are not applied, while baseline and cache controls continue. This is a numerical
gate, not statistical significance, and does not justify searching for another
direction after seeing test outcomes.

## Twenty-three functional arms per test input

The normal valid-candidate plan contains four main arms: `clean`, `pulse`,
`clean_add_shared` (clean plus `v32`) and `pulse_remove_shared` (pulse minus `v32`).
Add eight fixed norm-matched random
directions on clean and subtract the same eight on pulse: sixteen noise arms.
Their names are `clean_add_noise_k` and `pulse_remove_noise_k`, for `k=0..7`.
The vector is added once to residual output18 at offset32, after replaying the
appropriate cache through offset31 and processing the final bridge token.
Record both requested norm and realized FP32 displacement. Local MLP values
within block18 precede that residual patch; later blocks can respond to it.

Three further arms start from the pulse cache after offset8 and replace native
K **and** V using the clean cache at identical absolute positions, in layers15–27:

| Cache arm | Replaced positions | Question addressed |
| --- | --- | --- |
| Original-slot restoration | `L-1` only | Dependence on the original affected prompt slot |
| Later-slot restoration | `L` through `L+7` | Dependence on entries propagated during the first eight bridge tokens |
| Both restorations | Union of those positions | Technical full-source cut for this known intervention |

These arms are named `pulse_restore_prompt_cache`,
`pulse_restore_continuation_cache` and `pulse_restore_both_cache`, respectively.
If the shared candidate is invalid, the eighteen shared/noise endpoint arms and
two corresponding shared greedy arms per test input are explicitly `not_run`;
the two baselines and three cache arms remain planned measurements.

Then consume bridge tokens9–32 identically. The swap does not retroactively
change the logits already computed at offset8; only subsequent forwards are
new measurements. Never splice the whole clean cache under the name of a
single-slot intervention. The union arm is a technical control, not independent
evidence that a semantic mechanism has been isolated.

Native DynamicCache objects are cloned with independent tensor storage and
metadata. Donor and receiver snapshots must remain unchanged. Store per-layer
K/V hashes, replaced indices and untouched-slot checks, not public cache tensors.
Earlier prompt positions and KV layers0–14 should be unchanged by the query-only
pulse. At later bridge positions, KV15 is also an expected null; propagation can
begin at KV16. Full-prefix MATH rebuilds at offsets8 and32 check clean and pulse
against cached execution with explicit FP32 tolerances, not assumed bit equality.

## Functional outcomes and limits

Each test input has a fixed gold answer and a fixed foil. The primary outcome is
`sum log P(gold tokens | prefix) - sum log P(foil tokens | prefix)`, excluding EOS,
using separate cloned caches and teacher forcing over each complete alternative.
Report paired arm-minus-own-baseline margin changes for both insertion and
removal. Token-mean log-probability margins are secondary because gold/foil token
lengths can differ. Save token IDs, lengths and individual token probabilities;
first-token rank is not the probability or correctness of an entire answer.

Only the four main arms receive bounded greedy continuation on each of the eight
test inputs: at most 32 responses, each capped at 64 selected tokens including EOS.
Report EOS completion, truncation, exact output and format/semantic assessment
separately. Noise and cache arms have likelihood measurements but no free-answer
specificity result. The absence of their generated answers is explicit.

Report all eight test cases, both task families and all eight control directions.
Arms, tokens and control seeds do not increase the number of independent cases.
A persistent cache-mediated difference is ordinary model-state persistence; it
does not establish recognition of an observer, learned weight memory, a global
observer code, self-awareness, energy savings or a validated general bottleneck.
The Atlas experiment remains separate. No Atlas labels or coordinates are used.

## Run, checkpoint and export

Open the [small starter](../notebooks/temporal_functional_START.ipynb), select an
A100, run all, and supply the private `temporal_functional_paket.zip`. No Drive
mount is required. The [builder](../notebooks/build_temporal_functional.py)
embeds the reviewed sources; public notebooks are templates, not the private
reference/input package. Existing GPU Torch is retained and cached weights reused.

The runtime uses a 20-minute soft budget after model loading, with validated
checkpoints and explicit incomplete status. Installation, download/loading,
finishing a bounded operation and exporting can add time. This is a budget, not
a measured runtime promise. `paused` is not `complete`; no second GPU waits for
an earlier stage. Resume accepts only matching validated source/input checkpoints.

Download `temporal_functional_results.zip` for analysis and preserve
`temporal_functional_results_RAW.zip` for full data and resume. Compact output
is not a resume archive. On failure, save the offered diagnostic ZIP. End the
GPU session after saving the required files. CPU tests validate implementation
with small random native models; original-model findings require the actual run.
