# Native B14 → B18 local transport pilot

The earlier Qwen3-1.7B captures motivated two different anchors: a narrow middle
region with pooled participation rank 1.76 at block 14, and the strongest observed
pre-answer error-prediction AUROC at block 18 (0.820). The minimum of the pooled
rank was at block 13, not block 14, and the within-TriviaQA rank at block 14 was
16.98. These observations motivate a causal experiment; they do not establish a
unique attention bottleneck or a self-monitoring mechanism. Blocks are zero-based.

This pilot tests propagation and local transport with the original frozen
`Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`,
FP32, TF32 off, the original H286 coordinates and fixed REF-W statistics.
The primary observer-memory hypothesis remains open: an externally readable
perturbation trace alone does not show that the model represents or uses its own
processing condition. No training or checkpoint replacement takes place.

## Fixed inputs and comparisons

The private package binds 12 previously captured TriviaQA prompts and four
arithmetic prompts, using their original token IDs. Selection follows source
order with documented text/length exclusions and fixed answer curation. This
reuses exploratory data; it is not an independent held-out confirmation. The two
task families must be reported separately. Inputs, protocol, H286 files and all
executed source files are hashed before the run.

Each input has seven primary arms: clean/sham and query scaling after Q-norm
at block 13, 14 or 15 with temperatures 1.03 and 1.08. This changes the attention
distribution within the existing mechanism; it does not replace attention with
sliding-window attention. Residual outputs at 13/14/15 and 17/18/19 and the original
H286 measurements are captured. Block 18 is the fixed primary downstream readout;
17 and 19 are neighboring controls, not candidates for post-result reselection.

The pulse occurs only at the last original prompt token. Matched prefixes append
the same fixed gold continuation in every arm, inspecting offsets 0, 1, 4 and 8
when those tokens exist. Missing offsets remain explicitly unmeasured: no padding,
repetition or synthetic continuation is substituted. Gold tokens provide a
controlled input, not evidence of spontaneous memory. A separate branch generates
64 fresh greedy responses (clean and the three 1.08 arms for every input), each
ending at EOS or 64 selected tokens. Capped text is marked incomplete. Responses
are retained for later content assessment; they do not select prompts or doses.

## Actual local Jacobian and the Mem2Gen connection

For every prompt, the native Hugging Face suffix from residual output 14 through
output 18 defines `F(h14)`. All other source-token states are held fixed. The
adapter reuses the exact native masks, position IDs and rotary embeddings, and
executes only blocks 15–18. Autograd supplies JVPs and VJPs at the clean state;
no dense Jacobian or converted model is needed. MATH SDPA is used consistently
for this differentiable branch and its intervention comparisons. The primary
forward branch retains efficient SDPA. Baseline parity, finite-difference
agreement, uninstrumented/sham equality, unchanged weights and hook cleanup are
technical checks, not scientific outcomes.

For the block-14/1.08 pulse, compare `J delta_h14` with the actual `delta_h18`.
This quantifies local linear propagation. Projection onto this predicted
direction uses the observed target change to set its coefficient; it is a fitted
local diagnostic, not a held-out estimate of a mediator. Replay and removal of
the *entire* target delta are downstream cut controls: at the last prompt token,
near-perfect restoration can follow directly from the computational graph.

On the four fixed inputs 0, 1, 12 and 13, compute

```
delta14 = J.T @ inverse(J @ J.T + lambda**2 * I) @ delta18
```

with matrix-free conjugate gradients. This is the same Tikhonov right-inverse
principle used in [the upstream transport method](https://github.com/BurnyCoder/jacobian-mem2gen/blob/jacobian-transport/docs/jtransport.md),
applied here to a **local change between blocks 14 and 18**. The fixed damping is
0.1 times a four-iteration, seeded power estimate of gain; this is not an exact
singular value or a full spectral analysis. The bounded solver records convergence
and residuals. An unconverged approximate solution cannot be presented as an
accurate inverse.

Apply that source delta in a fresh clean forward pass and compare the achieved
block-18 change with the target. Controls copy the raw target delta directly
into block 14, and apply a fixed random direction with the transported delta's
source norm. Each control has one predetermined random direction per prompt;
this is not a random-seed distribution study. These tests connect native
attention interventions to actual Jacobian-based residual transport. They do
**not** fit or replicate the corpus-averaged final-layer J-lens, the complete
Mem2Gen self-patching grid, or its training experiment.

## Run and inspect

Open [the small starter](../notebooks/b14_b18_START.ipynb), choose an A100, run all
and upload the separately supplied `b14_b18_paket.zip`. No Drive mount is needed.
The source template contains no private reference bytes. Existing model weights
are reused and the installed GPU Torch is preserved. Installation and loading
precede a 45-minute compute budget, checked between complete request checkpoints.
An atomic request can overrun that soft limit; a paused run is not a completion.
The program has no idle-GPU wait stage.

Download `b14_b18_results.zip` for the analysis upload and also preserve
`b14_b18_results_RAW.zip`. The compact file retains all state and derivative
arrays and recorded metrics, but omits full-vocabulary logit arrays. It is
explicitly an analysis derivative, not an exact resume checkpoint. The RAW archive
retains every logit vector and the original checkpoints for independent KL
recomputation and resuming. Both retain source bindings and explicit missing
offsets, EOS/cap markers, technical checks, backend differences and solver residuals.
On failure, use the offered diagnostic archive. A failed compact export falls back
to the unchanged full archive.
The runtime records measured progress; this new derivative workload has not yet
been timed on the original A100 checkpoint. CPU tests use small random native
Qwen3 models to validate implementation and do not constitute experiment results.

After this stage, a processing-representation claim would still require an
independent test of generalization and functional use, beyond generic propagation
of a perturbation and beyond changed answer tokens. Persistent contextual state,
weight memory, passive observation and active attention intervention remain
distinct hypotheses.
