# Housing Price Predictor

Predicts sale prices for residential homes in Ames, Iowa, using the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset (1460 training rows, 80 features).

**Who'd use this / what "good" looks like:** a rough, fast price estimate for a home given its features — useful for a seller or agent sanity-checking a listing price. "Good" means predictions are typically within ~10% of the actual sale price, since being off by $10k matters far more on a $100k home than a $500k one — which is also why the model is trained on log(SalePrice) rather than raw dollars.

## Approach

1. **EDA** — examined the target distribution (right-skewed, skew ≈ 1.88) and missing values across 80 features.
2. **Log-transform the target** — `log(SalePrice)` brought skew down to ≈ 0.12, aligning the model's loss with proportional/percentage error rather than absolute dollar error (and matches the competition's own RMSE-on-log metric).
3. **Leakage-safe train/valid split** — done *before* any preprocessing, to avoid the imputer/encoder ever seeing validation data.
4. **Missing-value handling** — classified each column's missingness into two groups by checking `data_description.txt` and verifying edge cases empirically (e.g. `MasVnrType`):
   - **Group A** (feature genuinely doesn't exist, e.g. no pool/garage) → constant fill
   - **Group B** (genuinely unrecorded) → statistical fill (median/most-frequent), computed from training data only
5. **Encoding** — one-hot encoded all categorical columns (`handle_unknown='ignore'`, fit on training data only).
6. **Pipeline + ColumnTransformer** — repackaged the above into six column-group transformers (Group A/B/leftover × numeric/categorical), so preprocessing refits correctly and leak-free on any new split — necessary for cross-validation.
7. **Model comparison** — 5-fold cross-validated RandomForest vs XGBoost; also demonstrated XGBoost early stopping (stopped at 27/1000 rounds on a held-out split).
8. **Final model** — XGBoost, retrained on the full dataset, used to generate the competition submission.

## Results

| Model | CV RMSE (log SalePrice) |
|---|---|
| RandomForest | 0.1437 |
| XGBoost | **0.1397** |

XGBoost's cross-validated predictions also translate to a mean absolute error of ≈ $17,660 (≈ 10% average error) in actual dollar terms.

## Tech

Python, pandas, scikit-learn, XGBoost.

## Running it

```bash
# from the mlProjects root, with the shared .venv activated
cd housing
jupyter notebook housing.ipynb
```

Requires the Kaggle competition data (`train.csv`, `test.csv`, `data_description.txt`) placed in `housing/data/` — not included in this repo (gitignored per competition rules). Download from the [Kaggle competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).
