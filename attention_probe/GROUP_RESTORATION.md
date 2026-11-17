# Fixed H286 group restoration: exploratory A100 protocol

This protocol tests whether a small, fixed local H286 group carries part of the
effect of a mild attention intervention. It does not discover neuron communities,
train a model, or test memory formation or self-awareness.

## Model, sources, and fixed groups

- Model: `Qwen/Qwen3-1.7B`, revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Use FP32, TF32 disabled, SDPA/EFFICIENT_ATTENTION, evaluation mode, and no
  gradients or optimizer. Keep the original tokenizer/template and thinking off.
- Preserve the ordered H286 and fixed REF-W statistics from the prior experiment.
  G7 contains the **9 H286 coordinates in zero-based block 7**; G14 contains the
  **25 H286 coordinates in block 14**. No other H coordinates enter either patch.
- Within the already post-hoc H286 selection, membership of G7/G14 follows layer
  anatomy alone, without further selection by response, task success, activation
  size, or this experiment's result. Private neuron coordinate IDs are supplied
  by the existing reference bundle, not this file.
- The original 48-token archive SHA256 is
  `004eae9d743a529aa8e6661c11e892871a41fc6e43745310274a6ae524d432e2`.
  The 128-token follow-up archive SHA256 is
  `729a447424d1ed1df8bf49bde14cb3ad9933db02017e0ccd94a7bc623369ef65`.
  Preserve their source modules and results. New modules and the new plan have
  their own hashes; the 128-token source is a reference, not an editable template.

## Inputs and scope

`group_probe_items.json` contains ten records. Its first four original prompt,
target, category, ID, and pair fields are unchanged from `pilot_items.json`.
Preserve original request IDs r012–r015 and request seed `20261117 + index`.
New IDs r100–r105 must not be silently renumbered by iteration order.

| Requests | Family | Change from the original false-draft prompt |
|---|---|---|
| r012, r013 | p07, wug/color | Original correct and false drafts |
| r014, r015 | p08, exclusive switches | Original correct and false drafts |
| r100, r103 | p07, p08 | Remove the draft sentence only |
| r101, r104 | p07, p08 | Mark the same false draft explicitly untrusted |
| r102, r105 | p07, p08 | Keep the false draft; request a brief explanation |

The target facts remain green for p07 and B for p08. These are two known task
families, selected after earlier results. The variants do not form a held-out
confirmation set. Report originals and each variant separately; ten inputs,
350 arms, or 286 neurons must not be presented as independent task families.

## Intervention and donor definitions

The attention intervention is the existing single query pulse after q_norm at
the final prompt position: temperature T=1.08, beta=1/T, in block 7 or 14.
Use fresh prompt-only prefill and a fresh cache for every arm. No teacher-forced
gold answer enters free generation, and no cache is shared between arms.

For each prompt, compute clean/Sham and both pulse donors. Capture local signed
MLP gated activation `z = SiLU(gate_proj(x)) * up_proj(x)` immediately before
`down_proj`. Full local donor vectors may be read to construct the controls.
Never patch CETT, absolute z, zCETT, or a residual-stream coordinate instead.

For the tested block let `delta_G = z_pulse[G] - z_sham[G]`:

- **baseline_restoration**, pulse <- sham: on the pulsed recipient replace G by
  its clean donor values; equivalently subtract delta_G from those coordinates.
- **pulse_state_transfer**, sham <- pulse: on the clean recipient replace G by
  its pulsed donor values; equivalently add delta_G.
- **identity_clean** and **identity_pulse**: replace G by its own matching donor
  in the clean or pulsed recipient. Both must preserve the corresponding output.

For target and identity patches assign the donor FP32 values directly; the delta
formulas are algebraic descriptions, not a requirement to subtract/add with
extra rounding. Additive control displacements are recorded separately.

Patch only that block and that absolute prompt position. All other positions and
coordinates remain untouched. Do not clamp downstream H286 layers. The patch
and pulse hooks must be removed before cached decoding; downstream cache effects
then evolve naturally. Restoration means returning toward the baseline, which
can itself be wrong. It does not automatically mean repairing an answer.

## Controls and exact execution budget

For each block, use three fixed random masks outside H286, with the same number
of coordinates as G, and three fixed random directions supported inside G.
Fix masks, directions, seeds, sampling algorithm, and numerical guard constants
in the run plan before obtaining outcomes. Reuse them across all prompts.
Use the same mask/direction with opposite signs for restoration and transfer.
Each mask has unique coordinates; different seed masks may overlap, and their
overlap must be reported. Duplicate complete masks are rejected, not counted twice.

For an outside mask C, take the donor difference
`delta_C = z_pulse[C] - z_sham[C]` and scale it to the raw Euclidean norm of
delta_G. Scale each within-G random direction to that same norm. Apply the
negative control displacement to the pulsed recipient and the positive one to
the clean recipient. No seeds or masks may be replaced after inspecting results.
Degenerate source norms must yield an explicit invalid-control diagnostic,
not an unreported resample. A zero target displacement is an explicit no-op.

Record raw target/applied delta norms, source-control norm, scale factor, masks,
seeds, finite status, and achieved norm error. Also record the norm and cosine
of `W_down * delta` relative to the corresponding target patch. If a vector has
zero norm, its cosine is missing, not zero. Raw norm matching does not guarantee
equal MLP-output changes or equal functional damage.

| Arm type | Per block | Per input |
|---|---:|---:|
| Sham, block-14 pulse, block-7 pulse baselines | — | 3 |
| Target restoration and transfer | 2 | 4 |
| Clean and pulse identity patches | 2 | 4 |
| Three outside masks, both directions | 6 | 12 |
| Three within-G directions, both directions | 6 | 12 |
| Total prefill measurements | 16 per block + shared baselines | **35** |

Execute **350 prefill measurements** over the ten inputs. Free greedy decoding
is fixed in advance: seven main arms on each original input (three baselines,
G7/G14 restoration, G7/G14 transfer) give 28 answers; the three baselines on
each of six variants add 18. Thus **46 answers**, each at most **128 selected
tokens including EOS**. Log EOS but omit it from emitted text/tokens.
Do not force continuation after EOS. For every other arm explicitly store
generation status `not_run`; missing generation is not failure or empty output.

## Measurements and validation

- Compute full-vocabulary first-step `KL(reference || test)` on identical prompt
  tokens, using stable log-softmax and FP64 metric arithmetic. Reference names and
  direction must be explicit: report against both clean and same-block pulse
  when assessing restoration/transfer. Never compare full-run KL over differing
  free-generation histories as if their conditioning prefixes were identical.
- Save first-step top tokens, selected token, reference-token probabilities and
  ranks, and full-logit hashes. Target-specific contrasts use fixed alternatives
  across arms: green/brown or B/A. Tokenize both strings explicitly. A first-token
  contrast is not a whole-word or complete-answer probability for multi-token text.
- Report format/opening-token readouts separately from truth-versus-alternative
  readouts. Neither readout alone establishes semantic correction or awareness.
- Save pre/post-patch signed z, patch positions/masks, local delta diagnostics,
  and fixed-REF H286 readouts where available. Label the exact capture scope;
  prompt captures do not become decoder-state trajectories when generation grows.
- Check identity arms, a genuinely uninstrumented clean first forward, pulse and
  patch counts, hook cleanup, finite values, untouched coordinates/positions,
  original source hashes, frozen parameter evidence, and original baseline prefix
  compatibility. Numerical mismatch and token-prefix mismatch are separate fields.
- Keep semantic conclusion, strict surface match, instruction-format adherence,
  EOS/completion, and truncation as separate results. `noncanonical` is not a
  semantic-error label. In the two explanation-request variants it is also not
  automatically a format violation. Quoted drafts/premises do not count as the
  model's endorsed conclusion; a capped response has an unfinished final outcome.

## Interpretation and stopping

Primary hypotheses are conditional, local mediation: restoring G moves the
pulsed distribution toward the clean distribution, and transferring G moves the
clean distribution toward the pulsed one, beyond matched control effects.
Report all directions and variants, including failures. Control specificity here
is measured at prefill/logit level; no free control generations were requested.
The three seeds are descriptive controls, not three independent task samples.

A positive result is not the complete necessity/sufficiency test used in
[Goodfire's neuron-community paper](https://static.goodfire.ai/neuron-communities/neuron-communities-paper.pdf):
we do not remove G, retain G while ablating
all other MLP neurons, or test layer necessity. A negative local result does not
exclude redundancy, effects elsewhere in H286, or paths outside H286. No large
community search, new gradients, learned probes, or training is part of this run.

The notebook should run checks, donors, planned arms, save results and an archive,
then finish automatically. No GPU wait/poll stage or unattended queued follow-up.
Report measured wall time separately from setup; no energy claim without direct
energy telemetry. Avoid promising EOS or a fixed wall-clock finish time.
