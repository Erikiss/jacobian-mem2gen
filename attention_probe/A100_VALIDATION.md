# H286 A100 answer validation

This diagnostic follows the 2026-11-17 pilot. It reuses the same sixteen inputs
(eight correct/wrong external-draft pairs), original Qwen3-1.7B checkpoint,
ordered 286 coordinates, fixed REF-W statistics, FP32 arithmetic, and five arms.
It is exploratory and reuses discovery cases; it is not a held-out confirmation.

The pilot recorded teacher-forced target-prefix probabilities and activation
changes. Those probabilities did not identify the answers the model would
actually produce. This run retains that measurement and adds prompt-only greedy
generation to separate response formatting from the reported target score.

## Protocol

Each arm first performs the original teacher-forced measurement unchanged. A
separate forward pass then starts generation from `prompt_ids` only. No target
answer tokens enter the generation input. The attention intervention is applied
once at the last prompt token, after query normalization and before RoPE.

Generation uses a fresh KV cache for each arm. Intervention and capture hooks
close after the prefill. Subsequent decoding uses the returned cache and has no
further interventions. Effects may continue through cached representations;
this protocol does not isolate a memory or self-monitoring mechanism.

Generation is deterministic greedy decoding, bounded to 48 new tokens, with EOS
termination. Each step records its selected token and the ten highest scoring
token candidates. The generated text, token IDs, stopping reason, and any
truncation are retained. Explicit formatting rules classify exact or permitted
format-equivalent targets. Noncanonical text is kept for review, not automatically
treated as a semantically wrong answer. Token-limit termination remains visible.

The original teacher-forced arrays and the new prompt-prefill arrays are stored
separately. The generation capture covers the last four prompt tokens; the
original teacher-forced capture covers the last four total input tokens including
the supplied target. Absolute positions accompany the arrays. Generated decoding
steps are not represented by these prefill activation arrays.

## Controls and integrity

The five arms remain sham, T=1.03 at block 14, T=1.08 at block 14,
norm-matched random query perturbation at block 14, and T=1.08 at block 7.
Blocks are zero-indexed. Arm order is randomized with the original per-request
seed. Each request is saved atomically only after all five arms finish.

Technical checks must pass before the exploratory queue. They compare the sham
with an unperturbed baseline and verify once-only interventions and causal
activation invariants. Frozen-parameter identity/version evidence is retained.
The run plan binds the exact inputs, old and new helper code, generation settings,
and original model/reference pins. Resume accepts only matching plans and complete
request records with their bound technical-check evidence.

The wrapper preserves Colab's Torch/CUDA installation and installs the exact
Transformers and Accelerate versions in use by the original experiment. It runs
the helper in a fresh Python process. No Google Drive mount or external GPU
provisioning is required. There is no idle predecessor-waiting stage.

## Runtime and interpretation

The earlier pilot's recorded 53.97 seconds were preparation plus short
teacher-forced trials, not a benchmark of autoregressive generation. This run
prints a measured estimate after completed five-arm requests and checkpoints
when its run-time budget requires a pause. Setup and downloads are separate from
that experimental-loop estimate. The A100 is sufficient for this model size.

This test evaluates response strings and controlled state changes. It does not
locate a general attention bottleneck, measure energy savings, run Atlas weights,
or demonstrate the model recognizing or memorizing its own intervention.

The user-facing entry point is the small `h286_A100_START.ipynb` launcher with
the separately supplied private `h286_a100_paket.zip`. Public source templates
contain no private H286 or reference bytes.
