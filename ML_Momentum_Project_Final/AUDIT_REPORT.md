> **Current final verification (2026-11-03):** The previous audit below is retained as historical evidence, not current results. All five notebooks use 392 valid rows (274/59/59); their source code and math were independently inspected, notebooks re-executed, and current results verified against the comparison CSV. The only confirmed notebook error was stale plot legends stating 232/50/50; these have been changed to dynamic split counts. No optimizer equations or data preprocessing were modified. RMSprop applies momentum to the preconditioned gradient; Adam is standard Adam with first-moment momentum and bias correction. Equal epochs do not imply equal gradient updates; no no-momentum baseline is provided. Degree 2 is a predefined choice rather than an empirically selected degree.

> **Post-audit correction:** This historical audit predates the 392-row correction. Its old 332-row counts and performance metrics are superseded by README.md and results/optimizer_comparison.csv.

# Independent audit — 2026-11-03

## Results
- Batch GD with Momentum: validation MSE 22.4512; test MSE 18.5065; test RMSE 4.3019; test MAE 3.2778; test R² 0.6872.
- SGD with Momentum: validation MSE 23.5634; test MSE 20.8531; test RMSE 4.5665; test MAE 3.4524; test R² 0.6475.
- Mini-batch GD with Momentum: validation MSE 22.5010; test MSE 18.4688; test RMSE 4.2975; test MAE 3.2826; test R² 0.6878.
- RMSprop with Momentum: validation MSE 22.5461; test MSE 18.5373; test RMSE 4.3055; test MAE 3.2902; test R² 0.6867.
- Adam: validation MSE 22.6007; test MSE 18.3583; test RMSE 4.2847; test MAE 3.2504; test R² 0.6897.

## Audit status
- PASS: Original Auto MPG dataset (398 records), continuous mpg target, horsepower predictor, quadratic features, training-only scaling, consistent splits, correct MSE gradient.
- PASS: Five distinct implementations: classical momentum with full/single/mini batches, RMSprop with normalized-gradient momentum, Adam with first/second moments and bias corrections.
- PASS: Five notebooks independently executed with saved numerical and graphical outputs; metrics and comparison CSV regenerated.
- PASS: MSE, RMSE, MAE, and R² evaluated on held-out test data.
- NEEDS REVIEW: Removing 60 identical (horsepower, mpg) pairs may remove distinct cars; preserve existing methodology but consider an alternative sensitivity analysis.
- NEEDS REVIEW: Fixed polynomial degree 2 lacks comparative degree selection; acceptable as predefined model, not as proven optimum.
- NEEDS REVIEW: Equal epochs are not equal parameter-update or time budgets; no repeated-seed or statistical-significance analysis.
- NEEDS REVIEW: SGD high learning rates yield non-finite losses and are excluded by selection; they do not count as converged runs.
- NEEDS REVIEW: No no-momentum baseline, so acceleration caused by momentum cannot be demonstrated from these experiments alone.
- NEEDS REVIEW: Adam has the lowest observed test MSE, but Batch GD has the lowest validation MSE; test results must not be used to select a model.
- Note: The original ZIP was not modified. Changes were limited to adding MAE, correcting unsupported conclusion language, README audit caveats, regenerated outputs and CSV, and this report.
