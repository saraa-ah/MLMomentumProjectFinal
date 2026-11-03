# Auto MPG Degree-2 Polynomial Regression with Momentum

## 1. Project Overview
This machine learning project predicts vehicle fuel efficiency (`mpg`, miles per gallon) from engine power output (`horsepower`) using a degree-2 Polynomial Regression model trained via manual NumPy implementations of five gradient-based optimization algorithms with **Momentum**.

All five algorithms share an identical preprocessing pipeline, feature transformation, objective loss function, dataset split, random seed, training budget (250 epochs), and evaluation methodology to support a controlled comparison. However, the optimizers have different numbers of updates per epoch, so equal epochs do not imply equal computational budgets.

---

## 2. Dataset Description & Location
- **Dataset Source:** [UCI Machine Learning Repository — Auto MPG Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/autompg-dataset)
- **Local File Path:** [`auto-mpg.csv`](auto-mpg.csv) (stored directly in the working directory; self-contained and portable).
- **Initial Shape:** 398 rows $\times$ 9 columns.
- **Cleaning & Preprocessing Pipeline:**
  1. **Feature Extraction:** Subsets only the predictor `horsepower` and the target `mpg`.
  2. **Missing Value Handling:** Missing horsepower values encoded as `"?"` are coerced to `NaN` and dropped (6 rows dropped $\rightarrow$ 392 rows).
  3. **Deduplication:** No `(horsepower, mpg)` pair deduplication is performed. Different vehicles can legitimately share the same predictor and target values. All **392 valid observations** are retained.
  4. **Dataset Splitting:** Partitioned reproducibly using `random_state=42` into:
     - **Training Set:** 274 samples (~70%)
     - **Validation Set:** 59 samples (~15%)
     - **Test Set:** 59 samples (~15%)
  5. **Polynomial Transformation & Scaling:**
     - Degree-2 polynomial expansion: $[x, x^2]$
     - StandardScaler fitted **strictly on the training split only** ($z_1, z_2$) to prevent data leakage.
     - Bias intercept column ($w_0$) prepended, producing design matrices $X \in \mathbb{R}^{m \times 3}$.

Model hypothesis:
$$\hat{y} = w_0 + w_1 z_1 + w_2 z_2 = X w$$

Loss function (Mean Squared Error):
$$\text{MSE}(w) = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i)^2 = \frac{1}{m} \|X w - y\|_2^2$$

Analytic Gradient:
$$\nabla_w \text{MSE}(w) = \frac{2}{m} X^T (X w - y)$$

---

## 3. Mathematical Principles of Momentum
Gradient descent updates parameters directly along the instantaneous negative gradient direction ($w_t = w_{t-1} - \eta g_t$). On ravines or ill-conditioned quadratic surfaces where curvature is steep across one dimension and shallow along another, standard gradient descent oscillates transversally with slow progress toward the optimum.

**Momentum** introduces a physical velocity vector $v_t$ that accumulates exponentially decaying past gradients:

1. **Velocity Update:**
   $$v_t = \beta_1 v_{t-1} + g_t$$
2. **Parameter Update:**
   $$w_t = w_{t-1} - \eta v_t$$

where:
- $\beta_1 \in [0, 1)$ is the momentum decay factor (set to $\beta_1 = 0.9$).
- $\eta$ is the learning rate.
- $g_t$ is the gradient evaluated over the batch/sample.
- Initial state: $w_0 = [0, 0, 0]^T$, $v_0 = [0, 0, 0]^T$.

### Physical Intuition & Advantages
- **Constructive Accumulation:** Along directions where gradients consistently point in the same direction, velocity accumulates up to a factor of $\frac{1}{1 - \beta_1} = 10\times$, rapidly accelerating progress through flat regions.
- **Destructive Interference:** Along oscillating directions where gradient signs alternate, consecutive terms cancel out, significantly reducing transversal oscillation and stabilizing updates.

---

## 4. The Five Momentum-Based Optimizers
The project organizes the five optimization algorithms into standalone, beginner-friendly Jupyter Notebooks:

### 1. Batch GD with Momentum ([`01_batch_gd.ipynb`](01_batch_gd.ipynb))
- **Update Frequency:** 1 full-batch update per epoch over all 274 training examples.
- **Equations:**
  $$g_t = \frac{2}{N} X_{\text{train}}^T (\hat{y}_{\text{train}} - y_{\text{train}})$$
  $$v_t = \beta_1 v_{t-1} + g_t$$
  $$w_t = w_{t-1} - \eta v_t$$

### 2. SGD with Momentum ([`02_sgd.ipynb`](02_sgd.ipynb))
- **Update Frequency:** 274 single-sample updates per epoch in shuffled order (`seed=42`).
- **Equations:**
  $$g_{t, i} = 2 x_i^T (\hat{y}_i - y_i)$$
  $$v_t = \beta_1 v_{t-1} + g_{t, i}$$
  $$w_t = w_{t-1} - \eta v_t$$

### 3. Mini-Batch GD with Momentum ([`03_mini_batch_gd.ipynb`](03_mini_batch_gd.ipynb))
- **Update Frequency:** 9 mini-batch updates per epoch (`batch_size=32`: 8 batches of 32, 1 final partial batch of 18).
- **Equations:**
  $$g_{t, B} = \frac{2}{|B|} X_B^T (\hat{y}_B - y_B)$$
  $$v_t = \beta_1 v_{t-1} + g_{t, B}$$
  $$w_t = w_{t-1} - \eta v_t$$

### 4. RMSprop with Momentum ([`04_rmsprop.ipynb`](04_rmsprop.ipynb))
- **Update Frequency:** 9 mini-batch updates per epoch (`batch_size=32`).
- **Equations:**
  $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_{t, B}^2$$
  $$b_t = \beta_1 b_{t-1} + \frac{g_{t, B}}{\sqrt{v_t} + \epsilon}$$
  $$w_t = w_{t-1} - \eta b_t$$
  with $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

### 5. Adam (Adaptive Moment Estimation) ([`05_adam.ipynb`](05_adam.ipynb))
- **Update Frequency:** 9 mini-batch updates per epoch (`batch_size=32`).
- **Equations:**
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_{t, B}$$
  $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_{t, B}^2$$
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
  $$w_t = w_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
  with $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

---

## 5. Project Directory Structure
```
working_copy/
│
├── 01_batch_gd.ipynb         # Full-Batch GD with Momentum notebook
├── 02_sgd.ipynb              # Stochastic GD with Momentum notebook
├── 03_mini_batch_gd.ipynb    # Mini-Batch GD with Momentum notebook
├── 04_rmsprop.ipynb          # RMSprop with Momentum notebook
├── 05_adam.ipynb             # Adam (Momentum + RMSprop) notebook
├── auto-mpg.csv              # Original Auto MPG dataset
├── requirements.txt          # Python package dependencies
├── README.md                 # Complete documentation and benchmarking report
└── results/
    ├── optimizer_comparison.csv  # Final benchmark comparison metrics
    └── execution_verification.json  # Independent execution verification
```

---

## 6. Installation & Execution

### Environment Setup
From within the project directory, install the required packages:
```bash
pip install -r requirements.txt
```

### Running the Notebooks
Launch the Jupyter interface and open any of the standalone notebooks:
```bash
jupyter notebook
```
Each notebook is completely self-contained, executes linearly from top to bottom, and contains all training logs, evaluation tables, loss learning curves, and regression fit visualizations.

---

## 7. Experimental Results & Benchmark Comparison

All five notebooks were re-executed with **392 valid observations** (274 train / 59 validation / 59 test). No valid `(horsepower, mpg)` pairs were removed. All models use the same degree-2 feature pipeline, split, and 250 epochs. Learning rates were selected using validation MSE only.

| Optimizer | Selected LR | Validation MSE | Test MSE | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| Batch GD with Momentum | 0.0500 | 21.6877 | 19.1591 | 4.3771 | 3.4965 | 0.6123 |
| SGD with Momentum | 0.0010 | 21.6178 | 18.1342 | 4.2584 | 3.3773 | 0.6331 |
| Mini-batch GD with Momentum | 0.0050 | 21.7147 | 19.2420 | 4.3866 | 3.5156 | 0.6107 |
| RMSprop with Momentum | 0.0050 | 21.4690 | 18.9324 | 4.3511 | 3.5146 | 0.6169 |
| Adam | 0.1000 | 21.5061 | 18.9974 | 4.3586 | 3.5124 | 0.6156 |

## 8. Interpretation and Scope

The model selected by the lowest validation MSE in this run is **RMSprop with Momentum**. Test results are reported for assessment, not model selection. All five optimizer implementations and their mathematical equations were preserved; only the invalid pair-based deduplication was removed. This changes the sample size, the split, and therefore the measured outcomes. The 250-epoch budget is not equal in gradient-update count across optimizers. No conclusion about momentum speedup versus no-momentum training or statistical superiority is supported by these experiments.

**Verification:** See `results/execution_verification.json` and the saved outputs in all five notebooks.
