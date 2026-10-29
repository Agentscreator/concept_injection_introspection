# Concept Injection Introspection

> **Built on Anthropic's work.** This is a replication of the concept injection experiment from Anthropic's [*Emergent Introspective Awareness in Large Language Models*](https://transformer-circuits.pub/2025/introspection/index.html) (Lindsey, 2025). The research question, method, and probe design all come from that paper. This repo applies them to a small open-source model.

A replication of Anthropic's concept injection introspection experiment on **Llama 3.2 3B Instruct** (4-bit quantized).

The original work asked: can a language model detect when a concept vector has been injected directly into its hidden states? This repo runs that question on an open-source model with a full parameter sweep, fixing six design flaws present in an earlier pilot.

---

## What This Does

1. **Builds concept vectors.** Extracts last-token activations at each swept layer for 20 concepts (ocean, betrayal, algorithm, etc.) and subtracts a neutral baseline mean from 50 common words. Vectors are extracted at the same layer they are injected into (fix #4).

2. **Injects via forward hook.** Registers a `register_forward_hook` on transformer layers that adds `strength × scale × concept_vector` to the residual stream, where `scale` is the local residual norm (fix #5: makes strength comparable across layers).

3. **Probes introspection.** Asks the model: *"Do you detect an injected thought in your current processing? If so, what is it about?"*

4. **Sweeps the parameter space.** 5 layers (about 40–90% depth) × 4 strengths (0.25, 0.5, 1.0, 2.0, norm-relative) × 20 concepts, plus varied control prompts = **400 injected + 400 control trials**.

5. **Scores responses.** Keyword regex for detection, substring match for naming (excluding concept-dominated outputs), unique-word ratio for coherence.

6. **Spin run.** Repeats one layer and strength setting (layer 19, strength 0.25) with 3 system-prompt warning levels, with a **matched control arm** (fix #6). Control trials use the same introspection question with injection off. **120 trials total**: 20 injected and 20 control per warning level.

---

## Results (v2)

**Naming.** When the output stayed coherent, the model named the injected concept in 53 of 150 trials (35.3%), compared to 0 of 340 coherent control trials. Across all 400 injected trials the rate is about 13%. All 20 concepts were named at least once. This is consistent with plain steering. I have not yet run a steering control, so I cannot say whether naming is more than the push leaking into the text.

Naming rate on coherent injected trials, by layer and strength (coherent trials per cell in brackets):

| Layer (depth) | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|
| 11 (39%) | 0.75 (16) | 1.00 (11) | 0.00 (4) | 0.00 (2) |
| 15 (53%) | 0.45 (11) | 0.75 (12) | 0.00 (3) | 0.00 (2) |
| 19 (67%) | 0.07 (14) | 0.50 (12) | 0.00 (3) | 0.00 (2) |
| 22 (78%) | 0.00 (17) | 0.33 (12) | 1.00 (3) | n/a |
| 25 (89%) | 0.00 (15) | 0.18 (11) | n/a | n/a |

Cells at strengths 1.0 and 2.0 have between 0 and 4 coherent trials, so I would not read much into them. At the two strengths with enough trials (0.25 and 0.5), naming falls as layers get deeper, and it rises from 0.25 to 0.5 at every layer. Higher strengths break generation: about 73 of 100 trials stay coherent at 0.25, 58 at 0.5, 13 at 1.0, and 6 at 2.0.

![Heatmaps of naming rate and coherence by layer and strength](heatmaps_v2.png)

**Detection.** Not interpretable. In the matched spin run, the keyword check fired on 100% of control answers and 100% of injected answers at every warning level (20 trials per arm), so it cannot tell me whether the model notices injections. The regex matches words that appear in the question and system prompt (such as "detect" and "thought") and does not check whether the answer is yes or no.

| Warning level | Injected detection | Control detection |
|---|---|---|
| No warning | 1.00 | 1.00 |
| Mild warning | 1.00 | 1.00 |
| Strong warning | 1.00 | 1.00 |

### Caveats

- The sweep's control arm uses different prompts from the injected arm (a known confound, see Known Design Problems). Naming is not affected by this, but detection from the sweep is not interpretable.
- "Named" excludes outputs where the concept word exceeds 25% of tokens (fix #3), but the threshold is a blunt instrument.
- 4-bit quantization may affect activation geometry compared to full precision.
- The spin run used layer 19 at strength 0.25, where only 1 of 14 coherent trials named the concept. That setting barely shows the injection, so it is a weak place to test warning levels.

---

## Known Design Problems (not yet fixed)

1. **Detection measure is broken.** The keyword regex matches words from the question and system prompt and ignores whether the answer is yes or no. Both arms fire 100%, so detection cannot be measured. Planned fix: ask for a forced Yes or No and compare the model's probabilities, with the same prompt in both arms and a reversed version of the question.

2. **No steering control.** Naming is measured only while asking the introspection question. I have not injected the same vectors during an unrelated question to see how often the concept word appears anyway. Without that, naming cannot be separated from steering.

3. **Sweep control arm uses different prompts.** Control trials in the sweep use 20 varied neutral questions, while injected trials all use the same introspection question. This creates a vocabulary confound: the detection regex fires on words from the question even in negative answers. The matched spin run avoids this, and the sweep's detection numbers do not.

4. **Coherence selection bias.** Only 150 of 400 injected trials survived the coherence gate (unique-word ratio of at least 0.5). Survival is higher at lower strength, so the naming table is computed on a non-random subset of the parameter space.

5. **Response text is not included in the CSVs.** The raw model outputs are not in `sweep_results_v2.csv` or `spin_results_v2.csv`.

---

## Files

| File | Description |
|---|---|
| `concept_injection_kaggle.ipynb` | Full experiment notebook (Kaggle, T4 GPU × 2) |
| `sweep_results_v2.csv` | 800-row sweep results (400 injected + 400 control, by layer, strength, and concept) |
| `spin_results_v2.csv` | 120-trial warning-level spin run with matched controls |
| `heatmaps_v2.png` | Naming rate and coherence heatmaps by layer and strength |
| `writeup.md` | Results write-up with tables and analysis |

---

## Setup

Runs on Kaggle (free T4 × 2 GPU). Requires a Hugging Face token with access to `meta-llama/Llama-3.2-3B-Instruct`.

```python
# Store your HF token in Kaggle Secrets as HF_TOKEN
# Add-ons → Secrets → Add secret
from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
```

Open the notebook in Kaggle, enable GPU T4 × 2, enable internet, and run cells in order. The full sweep takes about 55 minutes on T4 × 2.

---

## Model

- **Llama 3.2 3B Instruct**, 28 layers
- **4-bit NF4 quantization** ([BitsAndBytesConfig](https://huggingface.co/docs/transformers/main_classes/quantization), float16 compute)
- Concept extraction at the same layer as injection (matched-layer design)
- **Layers swept**: 11, 15, 19, 22, 25 (about 39%, 53%, 67%, 78%, 89% depth)
- **Strengths swept**: 0.25, 0.5, 1.0, 2.0 (as a fraction of the local residual norm)

---

## Related Work

- Lindsey, J. (2025). [Emergent Introspective Awareness in Large Language Models](https://transformer-circuits.pub/2025/introspection/index.html). Anthropic. ([Anthropic summary](https://www.anthropic.com/research/introspection))
- Zou et al. (2023). [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)
- Turner et al. [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248)
