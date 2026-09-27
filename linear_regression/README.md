# Linear Regression from Scratch

Linear regression implemented with **NumPy only**, trained with gradient descent, and verified against scikit-learn. The goal was to understand the mechanics behind every training loop: forward pass, loss, gradients, and parameter updates.

## What's inside

1. **Synthetic data**: `y = 3x + 2 + noise` (noise std = 1), 100 points, 80/20 train/test split with a fixed seed
2. **Gradient derivation** by hand (MSE loss, chain rule)
3. **Gradient descent loop** in NumPy
4. **Learning-rate experiments**
5. **Multiple features + standardization**
6. **Mini-batch gradient descent**
7. **Comparison with sklearn** `LinearRegression`
8. **Test evaluation** with RMSE

## Gradients

With error $e_i = w x_i + b - y_i$ and loss $L = \frac{1}{n}\sum e_i^2$:

$$\frac{\partial L}{\partial w} = \frac{2}{n}\sum e_i x_i \qquad \frac{\partial L}{\partial b} = \frac{2}{n}\sum e_i$$

## Key results

**Learning rate (1000 epochs):**

| lr | final loss | result |
|---|---|---|
| 0.0001 | 1.41 | too slow |
| 0.001 | 1.15 | still converging |
| 0.01 | 0.96 | converged |
| 0.05 | nan | diverged |

**Feature scaling (3 features on scales 0–10, 0–1000, 0–1):**

| setup | final loss |
|---|---|
| unscaled, lr 0.01 | nan (diverged) |
| unscaled, lr 1e-6 | 125.6 (stuck) |
| standardized, lr 0.01 | 0.90 |

Converted back to original units, the standardized model recovered w ≈ [2.94, 0.50, −4.19] and b ≈ 2.27 (true: [3, 0.5, −4] and 2).

**Batch size (100 epochs, standardized data):** batch 1 and 16 converged (loss ≈ 0.90); full batch (80) was still at 1700, because it gets only 1 update per epoch.

**Comparison with sklearn (1 feature):**

| method | w | b |
|---|---|---|
| gradient descent, 10000 epochs | 2.98536 | 2.09458 |
| sklearn (closed-form) | 2.98536 | 2.09458 |

**Test RMSE: 0.948.** On unseen data, predictions are off by about 0.95 on average, roughly equal to the noise in the data, so the model captured all of the learnable pattern.

## Lessons learned

- Too small a learning rate wastes epochs; too large diverges to `nan`. The best is the largest one that still converges.
- Features on different scales break gradient descent; standardization (using **train** statistics only) fixes it.
- Mini-batches give more updates per epoch for the same compute.
- A flat loss curve doesn't guarantee converged parameters; comparing against an exact solution caught this.

## How to run

```bash
pip install numpy matplotlib scikit-learn
jupyter notebook linear_regression_from_scratch.ipynb
```

Python 3.9. All randomness uses `np.random.default_rng(42)`, so results are reproducible.