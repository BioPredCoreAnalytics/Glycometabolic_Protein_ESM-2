# 🧬 GlycoPred: A Supervised & Deep Learning Framework for Predicting Glycometabolic Proteins

**GlycoPred** is an advanced computational framework that integrates **classical machine learning**, **deep learning**, and **protein language model (PLM) embeddings** to identify and classify glycometabolism-associated proteins from raw sequence data. The framework combines handcrafted statistical-moment descriptors with contextualized embeddings from **ESM-2 (Evolutionary Scale Modeling)** to capture both explicit positional/compositional patterns and deep evolutionary/structural context learned from large-scale protein corpora.

> 📄 Manuscript: *"GlycoPred: A Supervised Learning Framework for Predicting Glycometabolic Proteins Using Proteomic Sequence Features"*

---

## 👥 Authors

| Name | Affiliation |
|---|---|
| **Hamza Shahab Awan** *(Corresponding Author)* | Department of Computer Science, COMSATS University Islamabad, Lahore Campus, Pakistan |
| Abid Sohail | Department of Computer Science, COMSATS University Islamabad, Lahore Campus, Pakistan |
| Tamim Alkhalifah | Department of Computer Engineering, College of Computer, Qassim University, Buraydah, Saudi Arabia |
| Fahad Alturise | Department of Cybersecurity, College of Computer, Qassim University, Buraydah, Saudi Arabia |
| Yaser Daanial Khan | Department of Computer Science, School of Systems and Technology, University of Management and Technology, Lahore, Pakistan |

📧 Correspondence: `hamzashahabawan@gmail.com` | `sp24-pcs-006@cuilahore.edu.pk`

---

## 🚀 What's New — Advanced ESM-2 & Deep Learning Extension

This version of GlycoPred extends the original statistical-moment-based pipeline with a **hybrid feature representation** and **state-of-the-art deep learning backbone**:

- 🧠 **ESM-2 Protein Language Model Embeddings** — Per-residue and per-sequence embeddings extracted from Meta AI's `esm2_t33_650M_UR50D` (or lighter `esm2_t12_35M_UR50D` for low-resource settings), capturing deep evolutionary, structural, and functional context far beyond handcrafted descriptors.
- 🔗 **Hybrid Feature Fusion** — Statistical-moment descriptors (Raw, Central, Hahn Moments, PRIM, RPRIM, AAPIV, RAAPIV) are concatenated/fused with ESM-2 embeddings via a learnable projection layer before classification.
- 🏗️ **Advanced Deep Learning Architectures**:
  - **Transformer Encoder Classifier** — multi-head self-attention over ESM-2 residue embeddings.
  - **CNN–BiLSTM–Attention Hybrid** — convolutional motif extraction + bidirectional sequence modeling + attention pooling.
  - **Attention-based Fusion Network** — cross-attention between statistical-moment features and ESM-2 embeddings for adaptive feature weighting.
  - Legacy baselines retained: **SVM, Random Forest, Extra Trees, CNN, LSTM**.
- 📊 **Explainable AI (XAI)** — SHAP-based attribution and attention-weight visualization to interpret which residues/features drive glycometabolic classification.

---

## 🗂️ Repository Structure

```
GlycoPred/
├── feature_extraction/
│   ├── statistical_moments.py      # Raw, Central, Hahn Moments
│   ├── position_descriptors.py     # PRIM, RPRIM, AAPIV, RAAPIV
│   └── esm2_embeddings.py          # ESM-2 embedding extraction (NEW)
├── model_training.py                # ML + DL model training (SVM, RF, ET, CNN, LSTM)
├── advanced_models.py                # Transformer / CNN-BiLSTM-Attention / Fusion Network (NEW)
├── evaluation/
│   ├── cross_validation.py
│   ├── independent_test.py
│   └── explainability.py            # SHAP + attention visualization
├── data/                             # Sequence data (access on request)
├── results/                          # Metrics, confusion matrices, plots
├── requirements.txt
└── README.md
```

---

## 🧩 Methodology Overview

### 1️⃣ Feature Extraction
Two complementary feature families are extracted from each protein sequence:

**A. Statistical Moment-Based Descriptors** (original framework)
- Raw Moments, Central Moments, Hahn Moments
- PRIM (Position Relative Incidence Matrix), RPRIM (Reverse PRIM)
- AAPIV (Amino Acid Position Incidence Vector), RAAPIV (Reverse AAPIV)

**B. ESM-2 Deep Embeddings** (advanced extension)
- Sequences are tokenized and passed through a pretrained **ESM-2** model (Facebook/Meta AI, via the `fair-esm` or `transformers` library).
- Per-residue embeddings (shape: `L × 1280` for `esm2_t33_650M_UR50D`) are mean-pooled / attention-pooled to obtain a fixed-length sequence embedding.
- Embeddings are optionally fine-tuned (LoRA/partial unfreezing) on the glycometabolic classification task.

```python
import torch, esm

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

data = [("protein1", sequence)]
_, _, batch_tokens = batch_converter(data)

with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    embedding = results["representations"][33].mean(1)  # sequence-level embedding
```

### 2️⃣ Feature Fusion
Statistical-moment vectors and ESM-2 embeddings are fused using:
- **Concatenation + Dense projection**, or
- **Cross-attention fusion**, where statistical features act as queries attending over ESM-2 residue embeddings (keys/values).

### 3️⃣ Model Development
| Category | Models |
|---|---|
| Classical ML | SVM, Random Forest, Extra Trees |
| Deep Learning (baseline) | CNN, LSTM |
| Deep Learning (advanced) | Transformer Encoder, CNN-BiLSTM-Attention, Attention Fusion Network |

### 4️⃣ Evaluation & Explainability
- 5-fold / 10-fold cross-validation
- Independent hold-out test set
- Metrics: Accuracy, Sensitivity, Specificity, MCC, AUC-ROC, F1-score
- Confusion matrix visualization
- SHAP value analysis + Transformer attention-map visualization for interpretability

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/GlycoPred.git
cd GlycoPred
pip install -r requirements.txt
```

**Key dependencies:**
```
fair-esm
torch
transformers
scikit-learn
tensorflow / keras
shap
numpy, pandas, matplotlib, seaborn
```

---

## ▶️ Usage

```bash
# Step 1: Extract statistical-moment features
python feature_extraction/statistical_moments.py --input data/sequences.fasta --output features/moments.csv

# Step 2: Extract ESM-2 embeddings
python feature_extraction/esm2_embeddings.py --input data/sequences.fasta --model esm2_t33_650M_UR50D --output features/esm2.npy

# Step 3: Train models (baseline + advanced)
python model_training.py --features features/moments.csv --labels data/labels.csv
python advanced_models.py --moments features/moments.csv --esm2 features/esm2.npy --architecture transformer

# Step 4: Evaluate
python evaluation/cross_validation.py --model results/best_model.pkl
python evaluation/explainability.py --model results/best_model.pkl
```

---

## 📊 Dataset

Curated glycometabolic protein sequences derived from **UniProt/Swiss-Prot** and **NCBI** repositories. Access available upon reasonable request — contact the corresponding author.

---

## 📜 Citation

If you use this framework, please cite:

```bibtex
@article{awan2025glycopred,
  title={GlycoPred: A Supervised Learning Framework for Predicting Glycometabolic Proteins Using Proteomic Sequence Features},
  author={Awan, Hamza Shahab and Sohail, Abid and Alkhalifah, Tamim and Alturise, Fahad and Khan, Yaser Daanial},
  journal={TBD},
  year={2025}
}
```

---

## 📬 Contact

For questions, collaboration, or data access requests:
**Hamza Shahab Awan** — `hamzashahabawan@gmail.com` / `sp24-pcs-006@cuilahore.edu.pk`

---

## 📄 License

This project is released for academic and research purposes. Please contact the authors regarding reuse or redistribution.
