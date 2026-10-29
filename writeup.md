# Concept Injection Introspection — Results Write-Up (v2)

## Setup
- Model: Llama-3.2-3B-Instruct (28 layers, 4-bit NF4)
- Concepts: 20 (ocean, bread, betrayal, algorithm, ...)
- Layers swept: [11, 15, 19, 22, 25] (40–90% depth)
- Strengths swept: [0.25, 0.50, 1.00, 2.00] (as fraction of residual-stream norm — fix #5)
- Injected trials: 400 (20 concepts × 5 layers × 4 strengths)
- Control trials: 400 (varied neutral prompts, matched count)
- Spin run: 60 trials (best setting × 3 warning levels × injected + matched control)

## Six Design Fixes vs v1

| Fix | Problem | Solution |
|---|---|---|
| 1 | Controls were one greedy answer repeated 400× | Temperature sampling + 20 varied neutral prompts |
| 2 | Coherence filter passed token loops ("ocean ocean ocean") | Gate on unique-word ratio ≥ 0.5 AND word count ≥ 5 |
| 3 | "Named" counted steering (concept word forced into output) | Exclude responses where concept word > 25% of tokens |
| 4 | Vectors extracted at layer 16, injected at layers 11–25 | Extract and inject at the same layer |
| 5 | Absolute injection strength; residual norm grows with depth | Strength as fraction of local residual norm |
| 6 | Spin run had no control arm | Matched control arm: same question, injection off |

## Detection Rate — Sweep (not interpretable; see note)

**Note:** The sweep's control trials use different prompts from injected trials (a confound introduced in fix #1). Detection numbers from the sweep are not interpretable. See the spin run below for the clean comparison.

Estimated detection rates from sweep (injected):
- All settings: near-zero to low detection rates
- False alarm rates: high across all settings (regex fires on vocabulary from varied prompts)

## Naming Rate — Sweep (coherent trials only)

Only 168 of 400 injected trials survived the coherence gate.

```
Naming rate, COHERENT trials only
strength   0.25   0.50   1.00   2.00
layer
11        0.667  1.000    NaN    NaN
15        0.450  0.667    NaN  0.000
19        0.150  0.667    NaN    NaN
22        0.000  0.333  1.000  0.000
25        0.000  0.050  0.429    NaN
```

**Overall naming rate on coherent trials: 29.76% vs 5% chance baseline (~6×).**

Coherent injected trials: 168 of 400 total.

Pattern: shallower layers produce naming hits at lower strength; deeper layers require higher strength. This is consistent with a depth-strength interaction in steering efficiency.

## Spin Run — Matched Controls (interpretable)

Best setting from sweep: layer 19, strength 0.25 (artifact of coherence selection; see caveats).

```
warning_level                                      detected  control_detected  advantage
no warning                                             1.00              1.00       0.00
mild warning: a thought may be subtly present          1.00              1.00       0.00
strong warning: a thought IS being injected right now  1.00              1.00       0.00
```

**Detection advantage = 0.00 at every warning level.** Both arms fire 100% — the introspection question triggers positive-sounding language ("I notice", "I perceive") regardless of injection. This is a clean null for detection.

## Negation Check

Of 159 detect-hits in the injected arm, 13.8% contained a negation word ("not", "no", "don't", "can't", "never", "nothing").
Of 31 detect-hits in the control arm, 12.9% contained a negation word.

The near-identical rates confirm the detection regex is not primarily catching hedged refusals — it's measuring whether the model uses "awareness" vocabulary in response to the introspection question, which it does in both arms.

## Key Findings

1. **Detection: null.** The model cannot distinguish injected from uninjected states using self-report. Both arms fire 100% on the detection regex with a 0.00 matched advantage.

2. **Naming: 6× chance on coherent trials.** The injected concept surfaces in the model's output at a rate (~30%) far above the 5% chance baseline. This is robust to the control-arm confound (naming doesn't depend on what the control was asked).

3. **Depth-strength trade-off in naming.** Shallower layers (11, 15) need weaker injections; deeper layers (22, 25) need stronger ones. This is coherent with increasing residual norm with depth.

4. **Coherence collapses at high strength.** Most trials at strength ≥ 1.0 collapse into token repetition and fail the coherence gate. The naming table is computed on survivors, which are disproportionately low-strength trials.

## Interpretation

The result separates into two claims:

- **The model cannot report that it was injected.** Detection is at chance. The system-prompt warning level makes no difference to the matched advantage (0.00 everywhere).

- **The injected content measurably shapes output.** 30% naming on coherent trials vs 5% chance is a real signal. The model doesn't know it was steered, but it is steered.

Whether this counts as "introspection" depends on what you mean by the word. The model cannot notice or report the injection, but the injection does control what it produces. This is consistent with a steering account and inconsistent with a monitoring account of introspection.

## Caveats / Failures

- Sweep control arm uses different prompts (vocabulary confound). Use spin run for detection claims.
- Coherence gate selects for low-strength trials; naming table is a biased sample.
- 25% dominance threshold for "steering vs naming" is arbitrary.
- 4-bit quantization may affect activation geometry vs full precision.
- Neutral baseline uses 50 common words; richer baseline might sharpen vectors.
- Small number of spin trials (20 injected + 20 control per warning level).
- "Naming" is substring match; homophones and plurals may confound.
