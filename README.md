# Concept Injection Introspection

A replication of Anthropic's concept injection introspection experiment on Llama 3.2 3B Instruct (4-bit quantized).

The original work asked: can a language model detect when a concept vector has been injected directly into its hidden states? This repo runs that question on an open-source model with a full parameter sweep.

## What This Does

1. **Builds concept vectors** — extracts last-token activations at layer 16 for 20 concepts (ocean, betrayal, algorithm, etc.), subtracts a neutral baseline mean from 50 common words.
2. **Injects via forward hook** — registers a `register_forward_hook` on transformer layers that adds `strength × concept_vector` to the residual stream at a chosen layer during generation.
3. **Probes introspection** — asks the model: "Do you detect an injected thought in your current processing? If so, what is it about?"
4. **Sweeps the parameter space** — 5 layers (40–90% depth) × 4 strengths (5, 15, 30, 60) × 20 concepts + 20 controls = 800 trials.
5. **Scores responses** — keyword regex for detection, substring match for naming, word count for coherence. Computes d-prime (detection rate − false alarm rate).
6. **Spin run** — repeats the best layer/strength setting with 3 system-prompt warning levels.

## Results

**The model could not detect concept vector injections.**

| Metric | Result |
|--------|--------|
| Detection rate (injected) | 0–5% across all settings |
| False alarm rate (control) | 100% everywhere |
| d-prime (best) | −0.950 (layer 19, strength 5.0) |
| Naming rate (best) | 10% (layer 11/15/19, strength 5.0) |
| Warning level effect | None — 5% detection at all 3 levels |

The core failure mode was a **polarity inversion**: control trials (no injection) almost always produced detection keywords ("I sense...", "I notice...") as part of the model's default hedging style, while injected trials at strength ≥ 15 caused generation to collapse into repetitive token loops ("defined defined defined...", "wikipedia wikipedia...") that didn't contain those keywords at all.

At strength = 5, injection was too weak to surface into the model's introspective output. At strength ≥ 15, generation broke before the model could report anything.

See `writeup.md` for full tables and caveats.

## Files

| File | Description |
|------|-------------|
| `concept_injection_introspection.ipynb` | Full experiment notebook (Colab, T4 GPU) |
| `sweep_results.csv` | 800-trial sweep results (layer × strength × concept) |
| `spin_results.csv` | 60-trial warning-level spin run results |
| `heatmaps.png` | Detection rate, false alarm rate, d-prime, naming rate heatmaps |
| `writeup.md` | Results write-up with tables and failure analysis |

## Setup

Runs on Google Colab (free T4). Requires a Hugging Face token with access to `meta-llama/Llama-3.2-3B-Instruct`.

```python
# Store your HF token in Colab Secrets as HF_TOKEN
# Never paste it directly into a cell
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')
```

Open the notebook in Colab and run cells in order. The full sweep takes ~55 minutes on a T4.

## Model

- **Llama 3.2 3B Instruct** — 28 layers
- 4-bit NF4 quantization (BitsAndBytesConfig, float16 compute)
- Concept extraction at layer 16 (~60% depth)
- Layers swept: 11, 15, 19, 22, 25 (40%, 55%, 70%, 80%, 90%)

## Related Work

- Anthropic's original concept injection / introspection experiments
- Representation Engineering (Zou et al., 2023)
- Steering vectors / activation addition
