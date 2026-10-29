# Concept Injection Introspection

> **Built on Anthropic's work.** This is a replication of the concept injection experiment from Anthropic's [*Emergent Introspective Awareness in Large Language Models*](https://transformer-circuits.pub/2025/introspection/index.html) (Lindsey, 2025). The research question, method, and probe design all come from that paper. This repo applies them to a small open-source model.

A replication of Anthropic's concept injection introspection experiment on **Llama 3.2 3B Instruct** (4-bit quantized).

The original work asked: can a language model detect when a concept vector has been injected directly into its hidden states? This repo runs that question on an open-source model with a full parameter sweep, fixing six design flaws present in an earlier pilot.

---

## What This Does

1. **Builds concept vectors** — extracts last-token activations at each swept layer for 20 concepts (ocean, betrayal, algorithm, etc.), subtracts a neutral baseline mean from 50 common words. Vectors are extracted at the same layer they are injected into (fix #4).

2. **Injects via forward hook** — registers a `register_forward_hook` on transformer layers that adds `strength × scale × concept_vector` to the residual stream, where `scale` is the local residual norm (fix #5: makes strength comparable across layers).

3. **Probes introspection** — asks the model: *"Do you detect an injected thought in your current processing? If so, what is it about?"*

4. **Sweeps the parameter space** — 5 layers (40–90% depth) × 4 strengths (0.25, 0.5, 1.0, 2.0, norm-relative) × 20 concepts + varied control prompts = **400 injected + 400 control trials**.

5. **Scores responses** — keyword regex for detection, substring match for naming (excluding concept-dominated outputs), unique-word ratio for coherence.

6. **Spin run** — repeats layer 19 at strength 0.25 with 3 system-prompt warning levels, with a **matched control arm** (fix #6): control trials use the same introspection question with injection off.

---

## Results (v2)

**Naming.** When the output stayed coherent, the model named the injected concept in 53 of 150 trials (35.3%), compared to 0 of 340 coherent control trials. Across all 400 injected trials the rate is about 13%. This is consistent with plain steering. I have not yet run a steering control.

| Layer | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|
| 11 (39%) | 0.75 (16) | 1.00 (11) | 0.00 (4) | 0.00 (2) |
| 15 (53%) | 0.45 (11) | 0.75 (12) | 0.00 (3) | 0.00 (2) |
| 19 (67%) | 0.07 (14) | 0.50 (12) | 0.00 (3) | 0.00 (2) |
| 22 (78%) | 0.00 (17) | 0.33 (12) | 1.00 (3) | n/a |
| 25 (89%) | 0.00 (15) | 0.18 (11) | n/a | n/a |

Numbers in brackets are coherent trials per cell. Cells at 1.0 and 2.0 have very few trials, so I would not read much into them.

**Detection.** Not interpretable. The keyword check fired on 100% of control and 100% of injected answers at every warning level (20 trials per arm), so it cannot tell me whether the model notices injections.

![Heatmaps showing naming rate across layers and strengths for coherent trials](heatmaps_v2.png)

---

## Known Design Problems (not yet fixed)

1. **Sweep control arm uses different prompts.** Control trials in the sweep use 20 varied neutral questions while injected trials all use the same introspection question. This creates a vocabulary confound: the detection regex fires on words from the control question ("detect", "thought") even in negative answers. The matched spin run avoids this; the sweep's detection numbers do not.

2. **Coherence selection bias.** Only 150/400 injected trials survived the coherence gate. Survival is higher at lower strength, so the naming table is computed on a non-random subset of the parameter space.

---

## Files

| File | Description |
|---|---|
| `concept_injection_kaggle.ipynb` | Full experiment notebook (Kaggle, T4 GPU × 2) |
| `sweep_results_v2.csv` | 800-trial sweep results (layer × strength × concept × injected/control) |
| `spin_results_v2.csv` | 120-trial warning-level spin run with matched controls |
| `heatmaps_v2.png` | Naming rate and coherence heatmaps by layer × strength |
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

Open the notebook in Kaggle, enable GPU T4 × 2, enable internet, and run cells in order. The full sweep takes ~55 minutes on T4 × 2.

---

## Model

- **Llama 3.2 3B Instruct** — 28 layers
- **4-bit NF4 quantization** (BitsAndBytesConfig, float16 compute)
- Concept extraction at the same layer as injection (matched-layer design)
- Layers swept: 11, 15, 19, 22, 25 (40%, 55%, 70%, 80%, 90% depth)
- Strengths swept: 0.25, 0.5, 1.0, 2.0 (as fraction of local residual norm)

---

## Related Work

- Lindsey, J. (2025). [Emergent Introspective Awareness in Large Language Models](https://transformer-circuits.pub/2025/introspection/index.html). Anthropic.
- [Representation Engineering (Zou et al., 2023)](https://arxiv.org/abs/2310.01405)
- [Steering vectors / activation addition](https://arxiv.org/abs/2308.10248)
