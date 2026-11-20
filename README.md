# 🎭 Sarcasm Detection in Text

![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-bb9af7?style=for-the-badge)
![Course](https://img.shields.io/badge/AIUB-Intro%20to%20Data%20Science-7aa2f7?style=for-the-badge)

A controlled comparison of **Bag-of-Words vs TF-IDF** text representations across **Naive Bayes, Logistic Regression and SVM** for detecting sarcasm, built in R on the **Sarcasm Corpus V2**.

> Most sarcasm-detection studies test a single representation in isolation. This project evaluates both representations with the same three classifiers on an identical stratified split, so performance differences can be attributed to the representation itself.

---

## 📊 Results

Held-out test set (N = 1,876), stratified 80/20 split.

| Representation | Classifier | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| BoW | Naive Bayes | 0.6812 | 0.6521 | **0.7772** | 0.7091 |
| BoW | **Logistic Regression** | 0.7020 | 0.6786 | 0.7676 | **0.7204** |
| BoW | SVM | 0.6898 | 0.6846 | 0.7036 | 0.6940 |
| TF-IDF | Naive Bayes | 0.6812 | 0.6521 | **0.7772** | 0.7091 |
| TF-IDF | Logistic Regression | 0.6839 | 0.6825 | 0.6876 | 0.6851 |
| TF-IDF | **SVM** | **0.7036** | **0.7242** | 0.6578 | 0.6894 |

**Key findings**
- 🏆 **BoW + Logistic Regression** gave the best balance overall (F1 = **0.7204**).
- 🎯 **TF-IDF + SVM** achieved the highest accuracy (0.7036) and precision (0.7242), at the cost of recall.
- 🔁 Naive Bayes scores identically under both representations, because its binary presence/absence encoding discards the weighting difference between BoW and TF-IDF.
- 📌 The effect of representation is **classifier-dependent**, not universal.

---

## 🗂️ Dataset

**[Sarcasm Corpus V2](https://nlds.soe.ucsc.edu/sarcasm2)** (Oraby et al., 2016): crowdsourced debate-forum responses labeled `sarc` / `notsarc` by human annotators. All three subsets are combined:

| File | Subset |
|---|---|
| `GEN-sarc-notsarc.csv` | General sarcasm |
| `HYP-sarc-notsarc.csv` | Hyperbole |
| `RQ-sarc-notsarc.csv` | Rhetorical questions |

---

## ⚙️ Pipeline

```
Raw text ─► Cleaning ─► Corpus ─► BoW / TF-IDF ─► 80/20 stratified split ─► NB · LR · SVM ─► Evaluation
```

1. **Preprocessing:** remove URLs and @mentions, convert to ASCII, lowercase, strip punctuation, numbers and English stopwords, drop documents left empty.
2. **Features:** Document-Term Matrices (terms appearing in ≥ 3 documents) as raw counts (BoW) and TF-IDF weights.
3. **Models**
   - **Naive Bayes** (`e1071`) on binarized presence/absence features
   - **Logistic Regression** (`glmnet`) with L1 regularization and 5-fold CV for λ, which keeps it stable on high-dimensional sparse features
   - **SVM** (`e1071`) with a linear kernel
4. **Evaluation:** confusion matrices, accuracy, precision, recall and F1 (`caret`), plus comparison plots (`ggplot2`).

---

## 🚀 How to Run

```r
install.packages(c("tm", "e1071", "caret", "glmnet", "slam", "ggplot2"))
```

```bash
Rscript ids_final_project_group_5.R
```

The script prints confusion matrices and the results table, and saves:
- `fig1_confusion_matrix.png`: confusion matrix for BoW + Logistic Regression
- `fig2_f1_comparison.png`: F1 comparison across all six configurations

---

## 📁 Repository Structure

```
├── GEN-sarc-notsarc.csv
├── HYP-sarc-notsarc.csv
├── RQ-sarc-notsarc.csv
├── ids_final_project_group_5.R      # full pipeline
└── IDS_final_project_group_5.pdf    # project report
```

---



📄 Full methodology, literature review and discussion are in the [project report](IDS_final_project_group_5.pdf).
