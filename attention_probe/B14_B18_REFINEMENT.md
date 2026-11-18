# B14 → B18: convergence and damping refinement

The original A100 pilot completed all 16 contexts and 64 free answers. Its native
Jacobian predicted the finite block-14 pulse at block 18 with about 3.4% median
relative error. The four regularized inverse solves were useful approximations,
but all reached their 16-iteration limit. Their actual target errors were 29–46%.
The raw copied-target control also had a larger source norm than the ridge solution.

This follow-up addresses those numerical and control limitations. It retains the
original frozen `Qwen/Qwen3-1.7B` revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, FP32, TF32 off, original prompt token IDs,
H286 coordinates and REF-W. Previous solver, runner and released notebooks remain
unchanged. All scientific sources and seven private input files are hash-bound.

## Fixed comparisons

There are four existing contexts, indices 0, 1, 12 and 13: two knowledge questions
and two arithmetic questions. These are reused exploratory cases, not a new
held-out evaluation. The source is decoder residual output 14 and the target is
residual output 18, both zero-based, at the last original prompt token. All other
source-token states are fixed in the native suffix map. Attention and derivative
comparisons use MATH SDPA consistently.

Fresh clean states, the pulse deltas and the JVP are compared to lightweight
anchors extracted from the verified original RAW archive. Their numerical gate is
`norm(current - reference) <= 1e-5 + 1e-3 * norm(reference)`; source identities and
hashes are exact. Accepted numerical drift is recorded and is not an assertion
that two GPU executions have exactly the same operator. The old 16-CG source
correction is retained for a direct replay comparison.

The target `b` is the freshly measured change at block 18 after the same mild
block-14 attention pulse. For the clean-state Jacobian `J`, solve

```
(J J.T + lambda**2 I) y = b
x = J.T y
```

The original absolute lambda for each context is frozen from its recorded run;
the power estimate is not fitted again. The stage order is:

| Phase | Lambda multiplier | Squared penalty multiplier | Contexts |
| --- | ---: | ---: | --- |
| Original damping, convergence check | 1.0 | 1.0 | All four first |
| Prespecified lower damping | 0.3 | 0.09 | All four next |
| Prespecified lower damping | 0.1 | 0.01 | All four last |

Each solve starts from zero, permits at most 512 total CG updates including
restarts, and requires a recomputed true dual relative residual of at most `1e-4`.
Native FP32 vectors and FP64 reductions are retained. A small recursive residual
is only a candidate for convergence. True residuals are checked every 16 updates,
at a candidate crossing and at finalization. Reliable residual checks can restart the
recurrence within the same iteration limit; tolerances and damping are never
relaxed to turn a failure into a success. Every finite approximation and termination
reason is reported separately from stage completion.

For every lambda and context, execute ten source interventions:

- the calculated ridge correction;
- the target direction copied to block 14 and scaled to the ridge source norm;
- eight fixed Gaussian directions, each scaled to that same source norm.

The eight directions use a local CPU generator and are reused across lambda
levels within each context. The first direction retains the prior noise seed;
no direction is selected or resampled from its effect. The prior-compatible
additive seed rule also shares seven directions between contexts 0/1 and 12/13:
18 distinct seeds supply the 32 context-direction uses. These shared directions
do not create independent replications across contexts. At the original lambda,
also replay the recorded 16-CG source correction and the original unnormalized
target copy. Thus there are 12 solves, 120 primary intervention measurements and
eight additional original-control measurements. Numerical and capture checks add
forward/derivative calls; these counts are not a count of all GPU operations.

## Outcomes and interpretation

Save the full source corrections, linear predictions `J x`, actual nonlinear
target changes, source/target norms, relative reconstruction errors and cosines.
Full-vocabulary first-token logits in the RAW archive support independent FP64
KL comparisons against both clean and pulse. H286 raw activations and fixed
reference normalizations are retained. Requested patch norm and the actual
post-addition source displacement are distinct because native addition rounds.

The primary comparison is numerical convergence at the original lambda. The
secondary comparison is the reconstruction-versus-source-norm tradeoff across
all three fixed damping levels. Smaller damping can increase source magnitude
and nonlinear departure even when linear reconstruction improves. Report every
context and level, including nonconvergence and missing stages; do not select a
best lambda and present it as an independent generalization result.

The previously recorded dual residuals imply a conditional exact-linear bound:
for `A = J J.T + lambda**2 I`, `x_hat = J.T y_hat`, `x_star = J.T A^-1 b`,
and `r = A y_hat - b`, exact adjointness gives
`norm(J x_hat - J x_star) <= norm(r)`. At the old lambdas this limits the possible
change in linear relative target error from solving further to about 0.44–1.34
percentage points. It motivates testing damping separately. It is not a certified
FP32, nonlinear endpoint, KL or answer-quality bound.

There are still only four contexts. Noise directions, lambda levels and
iterations are repeated technical comparisons. Eight noise directions do not
establish 95% superiority within an individual context. No free answers are
generated in this numerical stage, and first-token changes are not complete
answer improvements. These are context-local extensions of the regularized
transport idea in Mem2Gen, not a global J-lens fit or evidence that the model
recognizes, memorizes or uses an observer condition. Long matched trajectories,
cache/text controls and a held-out functional test remain separate next steps.

## Run and recover

Open [the small starter](../notebooks/b14_b18_refine_START.ipynb), select an A100,
run all and supply the private `b14_b18_refine_paket.zip`. No Drive mount is needed.
The existing GPU Torch is preserved; pinned Transformers/Accelerate are installed
and the model helper runs in a fresh process. Existing cached weights are reused.
The public notebook is a source template; private anchors and references are
present only in the separately supplied package.

All stages run automatically. There is a 20-minute soft compute budget after
model loading, checked between stages and cooperatively inside the solver.
Final residual calculation, checkpointing and export can overrun that boundary.
Installation and model download/loading are additional. A paused run is not
complete; no stage waits on another GPU. This new workload has not yet been timed
on the original A100 model. Prior pilot timings are evidence about the previous
workload, not a promised runtime for this one.

Atomic stage checkpoints bind the plan, sources, tensors and technical proofs.
A resume skips only validated completed stages; a diagnostic partial solve is
not silently reused as a warm start. Global completion requires all twelve stages
and separately reports how many solvers converged.

Download `b14_b18_refine_results.zip` for analysis and preserve the additional
`b14_b18_refine_results_RAW.zip`. The compact derivative retains state/derivative
arrays and metrics; RAW retains full logits and exact resume data. Only RAW is a
resume archive. On failure, save the offered diagnostic ZIP. Once the archives
are saved, the GPU session can be ended.

CPU tests verify implementation with explicit matrices and small native Qwen3
models. They do not supply new original-model scientific results.
