"""Evaluate the classifier on one local CSV file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from cluster_local_logistic import ClusterLocalLogisticClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV file")
    parser.add_argument("--target", required=True, help="Binary target column")
    parser.add_argument("--positive-label", default="1", help="Value treated as class 1")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max-clusters", type=int, default=20)
    parser.add_argument("--predictions", type=Path, help="Optional output CSV")
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    if args.target not in data:
        parser.error(f"Target column {args.target!r} is missing")
    raw_target = data.pop(args.target)
    if raw_target.isna().any() or raw_target.nunique() != 2:
        parser.error("Target must contain exactly two nonmissing values")
    labels = (raw_target.astype(str) == args.positive_label).astype(int)
    if labels.nunique() != 2:
        parser.error("--positive-label must match one of the two target values")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must lie between 0 and 1")

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=args.test_size,
        stratify=labels,
        random_state=args.seed,
    )
    model = ClusterLocalLogisticClassifier(
        max_clusters=args.max_clusters,
        random_state=args.seed,
    ).fit(x_train, y_train.to_numpy())
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    report = {
        "n_train": len(x_train),
        "n_test": len(x_test),
        "test_positive": int(y_test.sum()),
        "selected_k": model.selected_k_,
        "selected_C": model.selected_c_,
        "selected_imbalance_cutoff": model.selected_cutoff_,
        "f1": f1_score(y_test, prediction, zero_division=0),
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "clusters": model.cluster_report().to_dict(orient="records"),
    }
    print(json.dumps(report, indent=2))

    if args.predictions is not None:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "row_index": x_test.index,
                "y_true": y_test.to_numpy(),
                "y_pred": prediction,
                "positive_probability": probability,
            }
        ).to_csv(args.predictions, index=False)


if __name__ == "__main__":
    main()

