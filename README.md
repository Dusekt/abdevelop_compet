# 🧬 AbDevelop: Antibody Developability Prediction (Ginkgo 2025 Competition)

**Author:** [dusekt](https://huggingface.co/spaces/ginkgo-datapoints/abdev-leaderboard)  
**Competition:** [Ginkgo Antibody Developability Prediction Challenge 2025](https://huggingface.co/spaces/ginkgo-datapoints/abdev-leaderboard)  

---

## 📖 Overview

This repository contains my submission for the **2025 Ginkgo Bioworks Antibody Developability Prediction Competition**.

The goal is to predict experimental developability properties for antibody sequences:

| Task | Property | Index in Code |
|------|----------|----------------|
| Titer | Productivity | 0 |
| Thermostability | Tm2 | 1 |
| Self-association | DLS | 2 |
| Polyreactivity | PSR | 3 |
| Hydrophobicity | SMAC | 4 |


---

## 🧠 Model Architecture

### Base Models
Each base model is a small neural network trained on clustered cross-validation folds.
Pretrained models were used as inputs:

| Model | Input | Description |
|:--------|:-------|:-------------|
| **AbLang** | Sequence embeddings (AbLang) | Captures structural sequence representations |
| **DeepSP** | Predicted descriptors (DeepSP) | Encodes solvent and developability-related properties |

- Architecture: FC layers with dropout, separate head for AbLang and DeepSP descriptors  
- CV: Cluster-based (ensures non-overlapping antibody families) - predefined by the competition
- Each task → 4 models × 5 folds = **20 base models per property**

---

## Pretraining

Models were pretrained on the Therapeutic Antibody Profiler (TAP) properties computed on the TheraSabDab database.
Pretrainning tasks were treated as hyperparameters themselves and optimized for with **Optuna**.

 - see scripts for further details.

## Data

All data used are in the data directory.

 - all_data.npy - data provided by the competition (CV dataset)
 - fold_array.npy - specifying data splitting for cross validation (provided by the competition)
 - thera_!!_DSP.npy - DeepSP predictors predicted on the TheraSabDab dataset (https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/therasabdab/search/)
 - thera_11_embeddings2.npy - AbLang embeddings for the TheraSabDab database
 - thera_TAP.npy - properties predicted by the Therapeutic Antibody Profiler (https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/tap) on the TheraSabDab data
 - train_DSP_out.npy - train data - DeepSP descriptors (compet. dataset)
 - train_embeddings2.npy - train data - AbLang embeddings (compet. dataset)

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

## 📚 References

**Therapeutic Antibody Profiler (TAP)**  
M.I.J. Raybould,C. Marks,K. Krawczyk,B. Taddese,J. Nowak,A.P. Lewis,A. Bujotzek,J. Shi, & C.M. Deane,  Five computational developability guidelines for therapeutic antibody profiling, Proc. Natl. Acad. Sci. U.S.A. 116 (10) 4025-4030, https://doi.org/10.1073/pnas.1810576116 (2019).

https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/tap

**AbLang**  
Addressing the antibody germline bias and its effect on language models for improved antibody design
Tobias H. Olsen, Iain H. Moal, Charlotte M. Deane
bioRxiv 2024.02.02.578678; doi: https://doi.org/10.1101/2024.02.02.578678
Now published in Bioinformatics doi: 10.1093/bioinformatics/btae618
                                            
https://github.com/TobiasHeOl/AbLang2

**DeepSP**  
Wang B, Zhang X, Xu C, Han X, Wang Y, Situ C, Li Y, Guo X. DeepSP: A Deep Learning Framework for Spatial Proteomics. J Proteome Res. 2023 Jul 7;22(7):2186-2198. doi: 10.1021/acs.jproteome.2c00394. Epub 2023 Jun 14. PMID: 37314414.

https://github.com/Lailabcode/DeepSP
