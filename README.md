# CEFR Learner Writing Classification: T5Gemma, ModernBERT & Gemma 4
This repository is a companion to the Poster Presentation:

Cooper, C. R. (2026, September 8–11). *An exploration of local LLMs for the classification of English learner writing by CEFR level* [Poster presentation]. The 6th Asia Pacific Corpus Linguistics Conference (APCLC2026), Toyama, Japan.

Eight-level CEFR classification (A2, A2+, B1, B1+, B2, B2+, C1, C1+) comparing fine-tuned encoders against a prompted decoder-only model.

## Contents

### Data preparation

| File | Purpose |
|---|---|
| `WriteAndImprove_sampling_plus_levels.ipynb` | Builds the train / validation / test splits from the Write & Improve Corpus (see below for access details) |

### `encoder/` — fine-tuned models

T5Gemma 2 (4B) and ModernBERT-large (395M), each trained with Conditional Ordinal Regression for Neural Networks (CORN) loss and with standard (nominal) classification.

| File | Purpose |
|---|---|
| `t5gemma2_encoder_only.ipynb` | T5Gemma fine-tuning, both loss functions |
| `PROBABILITIES_t5gemma2_encoder_only.ipynb` | T5Gemma prediction probabilities and calibration |
| `PROBABILITIES_modernbert_cefr.ipynb` | ModernBERT fine-tuning, probabilities and calibration |
| `test_set_evaluation_T5Gemma_ModernBERT.ipynb` | Held-out test set evaluation, all four conditions |
| `model.py` | `T5Gemma2OrdinalClassifier` |
| `metrics.py` | Shared evaluation metrics (QWK, MAE, exact and adjacent accuracy, macro F1) |

### `decoder/` — prompted model

Gemma 4 31B Instruct (Q4_K_M) served locally via LM Studio. Zero-shot, few-shot (8, 16, 24) and conceptually Bayesian prompting.

| File | Purpose |
|---|---|
| `gemma4_cefr_classification_VAL.ipynb` | Validation set run |
| `gemma4_cefr_classification_TEST.ipynb` | Test set run |
| `metrics.py` | Shared evaluation metrics |

## Data

The Write & Improve Corpus 2024 is not included here. It is available from Cambridge University Press & Assessment:

https://researchdatasets.cambridge.org/datasets/write-and-improve-corpus-2024

Nicholls, D., Caines, A., & Buttery, P. (2024). *The Write & Improve Corpus 2024: Error-annotated and CEFR-labelled essays by learners of English.* Cambridge University Press & Assessment. https://doi.org/10.17863/CAM.112997

## Dependencies

Python 3.12, with:

```
torch
transformers>=5.12.1
coral-pytorch
scikit-learn
pandas
numpy
plotly
scipy
statsmodels
```

`transformers>=5.12.1` is required for T5Gemma 2 support. `coral-pytorch` provides the CORN loss and label functions. The decoder notebooks additionally require a local [LM Studio](https://lmstudio.ai/) server running Gemma 4 31B Instruct.

## Notes

Model checkpoints are not distributed.

Contact: cooper@waseda.jp
