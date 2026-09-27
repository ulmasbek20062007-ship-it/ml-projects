# ML Projects

A structured, self-directed roadmap for building real machine learning skills — 
from foundational classification problems through production-style ML engineering.

## Structure

Each project lives in its own subfolder with its own notebook and README.

**Level 1 — completed**
- `iris/` — Multiclass flower classification: baseline reasoning, tuned Logistic Regression, cross-validation, error analysis
- `housing/` — House price regression: leakage-safe split, imputation, Pipeline + ColumnTransformer, RandomForest vs XGBoost
- `linear_regression/` — Linear regression from scratch in NumPy: hand-derived gradients, gradient descent, learning-rate experiments, feature scaling, mini-batches, verified against sklearn

**Coming next**
- Level 2: Customer Churn (sklearn), Image Classification (PyTorch CNN), Sentiment Analysis (PyTorch), Transfer Learning (PyTorch), Neural Network from scratch (NumPy → PyTorch)
- Level 3: Real-time Fraud Detection, Automated ML Pipeline

## Roadmap

Progressing through five levels of ML project complexity:
1. **Level 1** — Clean data, local notebooks, core modeling fundamentals
2. **Level 2** — Messy real-world data, proper project structure, deep learning with PyTorch
3. **Level 3** — ML engineering: APIs, Docker, monitoring
4. **Level 4/5** — Cloud-scale systems and research-level work (future)

## Setup

The repo uses a shared virtual environment (`.venv`) at the root. See each 
project's README for the libraries it needs and how to run it.