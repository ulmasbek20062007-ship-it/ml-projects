# Iris Flower Classification

**Level 1 project** — first full modeling loop: load, clean, train, evaluate.

## Problem

Predict the species of an iris flower (setosa, versicolor, or virginica) from 
four measurements: sepal length, sepal width, petal length, and petal width.

## Dataset

The classic Iris dataset (150 samples, 3 balanced classes, no missing values), 
loaded via `sklearn.datasets.load_iris()`.

## Approach

1. **EDA** — Histograms and pairplots showed petal length/width separate the 
   three species far more cleanly than sepal measurements; setosa is trivially 
   separable, versicolor/virginica overlap more.
2. **Baseline** — With 3 evenly balanced classes, guessing one class every 
   time gives ~33% accuracy — the floor any real model needs to beat.
3. **Manual model** — A simple petal-length threshold rule, used as a 
   non-ML baseline before reaching for scikit-learn.
4. **Logistic Regression** — Trained and evaluated with 5-fold cross-validation, 
   then tuned across a range of `C` (regularization) values.
5. **Error analysis** — Visualized misclassified points against petal 
   length/width; errors concentrate at the versicolor/virginica boundary, 
   consistent with the EDA.
6. **Final evaluation** — Model retrained on the full training set and 
   evaluated once on a held-out test set, with a fixed `random_state` for 
   reproducibility.

## Results

- Manual (petal-length threshold) model: **94.64%** accuracy
- 5-fold cross-validation (tuned, `C=1`): **96.36%** mean accuracy
- Final test set (38 held-out samples): **100%** accuracy

**Note:** the 100% test result is on a small 38-sample holdout — a strong 
result, but not proof the model is flawless. Iris is a highly separable 
dataset, so a small test set can easily land on a perfect score by chance.

## Lessons learned

- Without a fixed `random_state`, `train_test_split` produces a different 
  split — and therefore a different reported accuracy — every run. Fixed 
  early to make results reproducible.
- Cross-validation accuracy and true held-out test accuracy measure different 
  things; don't conflate them when reporting results.
- A variable named `X_train` doesn't guarantee it was built from the correct 
  training split — always verify shapes after any split/filter operation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install scikit-learn pandas numpy matplotlib seaborn jupyter
```