# Cluster-local logistic regression

This repository contains a standalone implementation of a binary classifier that routes observations through K-means++ clusters. A cluster with one training class returns that class. Every mixed cluster fits its own logistic regression model. A mixed cluster receives class weights when its minority-class share is below a cutoff selected by training-only validation.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
python -m pip install -e .
```

The package uses scikit-learn's [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html), and logistic regression implementations.

## Quick start

```python
import pandas as pd
from cluster_local_logistic import ClusterLocalLogisticClassifier

X_train = pd.DataFrame(
    {
        "income": [24, 27, 31, 35, 60, 63, 68, 72, 75, 80],
        "group": ["A", "A", "B", "B", "A", "A", "B", "B", "A", "B"],
    }
)
y_train = [0, 0, 0, 1, 0, 1, 1, 1, 1, 0]

model = ClusterLocalLogisticClassifier(
    max_clusters=3,
    c_values=(0.1, 1.0, 10.0),
    imbalance_cutoffs=(0.1, 0.2, 0.3, 0.4, 0.5),
    random_state=101,
)
model.fit(X_train, y_train)

X_new = pd.DataFrame({"income": [29, 70], "group": ["A", "B"]})
print(model.predict(X_new))
print(model.predict_proba(X_new)[:, 1])
print(model.cluster_report())
```

For a CSV file with a binary target:

```bash
python examples/train_csv.py --csv data.csv --target outcome --positive-label 1
```

The example script makes a stratified 80/20 split, fits the classifier on the training portion, and prints held-out F1, accuracy, precision, and recall. It can optionally save test predictions with `--predictions predictions.csv`. Dataset-specific row filtering and removal of ID columns should be performed before fitting.

## Method

1. Fit numeric median imputation and categorical one-hot encoding on training rows only. Missing categorical values become a distinct category. An unknown category at prediction time is ignored by the fitted encoder.
2. Compute a training-only within-cluster sum-of-squares (WCSS) curve. The selected cluster count is the interior point with the largest normalized gap from the line connecting the curve endpoints. The curve evaluates one extra endpoint so `max_clusters` remains selectable. At most 20,000 training rows are used for this step by default.
3. Use two stratified internal validation splits to choose the local logistic regularization parameter `C` and the minority-share cutoff by mean F1. Ties favor the smaller `C`, then the smaller cutoff. Encoders and imputers are refitted inside each validation split.
4. Fit K-means++ on all training rows with the selected cluster count. A cluster with exactly one class stores that label. A mixed cluster fits a logistic regression model, using `class_weight="balanced"` only when its minority-class share is strictly below the selected cutoff.
5. Route each new observation to its nearest centroid. The assigned cluster supplies the positive-class probability. The predicted class is 1 when the probability is at least 0.50.

The K-means++ gate uses imputed and encoded numeric features **without scaling**. Each mixed cluster's logistic model fits its own `StandardScaler` on that cluster's training rows. This matches the preprocessing used by this implementation. Euclidean clustering can be sensitive to feature units, so document any different scaling protocol if you change it.

`fit` requires a pandas DataFrame with unique string column names and a target containing both classes encoded as `0` and `1`. `predict_proba` returns two columns in class order `[0, 1]`. The fitted estimator exposes `selected_k_`, `selected_c_`, `selected_cutoff_`, `wcss_curve_`, `validation_results_`, and `cluster_report()`.

## Reproducibility scope

This package provides the classifier and a held-out evaluation example. It does not bundle datasets, article tables, or the other comparison methods. Reproducing reported benchmark values additionally requires the same data files, row filtering, splits, random seeds, and comparison implementations.

## Tests

```bash
python -m unittest discover -s tests -v
```

