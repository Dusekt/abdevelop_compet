# 🧬 AbDevelop: Antibody Developability Prediction (Ginkgo 2025 Competition)

**Author:** [dusekt](https://huggingface.co/spaces/ginkgo-datapoints/abdev-leaderboard)  
**Competition:** [Ginkgo Antibody Developability Prediction Challenge 2025](https://huggingface.co/spaces/ginkgo-datapoints/abdev-leaderboard)  
**Category:** Open-Source Submission 🏆  

---

## 📖 Overview

This repository contains my submission for the **2025 Ginkgo Bioworks Antibody Developability Prediction Competition**.

The goal is to predict experimental developability properties for antibody sequences:
- **Hydrophobicity** task 4 in the code 
- **Polyreactivity** task 3 in the code 
- **Self-association** task 2 in the code 
- **Thermostability (tm2)** task 1 in the code 
- **Titer** task 0 in the code 

---

## 🧠 Model Architecture

### Base Models
Each base model is a small neural network trained on clustered cross-validation folds:

| Branch | Input | Description |
|:--------|:-------|:-------------|
| **AbLang** | Sequence embeddings (AbLang) | Captures structural sequence representations |
| **DeepSP** | Predicted descriptors (DeepSP) | Encodes solvent and developability-related properties |

- Architecture: FC layers with dropout, separate head for AbLang and DeepSP descriptors  
- CV: Cluster-based (ensures non-overlapping antibody families) - predefined by the competition
- Each task → 4 models × 5 folds = **20 base models per property**

---

## Pretraining

Models were pretrained on the Therapeutic Antibody Profiler (TAP) properties computed on the TheraSabDab database and some other intern sequences. These data are **NOT** provided. Only dummy data, as i dont know if i can publish TAP data and the other intern sequences.

 - see scripts for further details.

## ⚙️ Meta-Model (Ensemble)

Predictions from all base models are stacked and used as meta-features.

The **meta-model** is trained via **5-fold Ridge Regression** optimized with **Optuna**:

```python
from abdevelop import EnsembleModel
# Instantiate for a specific task
model = EnsembleModel(task_id=2)
# Example input
N, E, D = 8, 480, 30
embeddings = np.random.randn(N, E)
descriptors = np.random.randn(N, D)

# Predict
y_pred = model(embeddings, descriptors)
print("Predictions:", y_pred.shape, y_pred[:5])
