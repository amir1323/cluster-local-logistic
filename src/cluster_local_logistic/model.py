"""K-means++ routing with local logistic experts.

The estimator accepts a pandas DataFrame and binary targets encoded as 0/1.
All fitted transformations are learned from the training data passed to fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


@dataclass
class _ClusterExpert:
    """One fitted cluster; exactly pure clusters have no logistic model."""

    mode: str
    n_samples: int
    n_positive: int
    minority_share: float
    pure_label: int | None
    model: Pipeline | None


class ClusterLocalLogisticClassifier(ClassifierMixin, BaseEstimator):
    """Binary classifier with K-means++ routing and local logistic regression.

    Parameters
    ----------
    max_clusters:
        Largest selectable cluster count. The WCSS curve evaluates one
        additional endpoint so that this value itself can be selected.
    c_values:
        Candidate inverse L2 regularization strengths for local models.
    imbalance_cutoffs:
        Candidate thresholds for the minority-class share within a cluster.
        A mixed cluster is weighted when its share is strictly below the
        selected cutoff.
    validation_splits:
        Number of stratified, training-only validation splits used to choose
        C and the imbalance cutoff.
    validation_size:
        Fraction of the fit data reserved for each internal validation split.
    elbow_sample_size:
        Maximum number of training rows used to calculate the WCSS curve.
    random_state:
        Integer seed controlling sampling, clustering, and validation.

    Notes
    -----
    Numeric missing values use training medians; missing categorical values
    become an explicit category. Categorical values are one-hot encoded from
    the training partition only. K-means++ uses this representation with
    numerical values on their original scale. Each mixed cluster's logistic
    model fits its own StandardScaler. Class labels use a fixed 0.50 cutoff.
    """

    def __init__(
        self,
        *,
        max_clusters: int = 20,
        c_values: tuple[float, ...] = (0.1, 1.0, 10.0),
        imbalance_cutoffs: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5),
        validation_splits: int = 2,
        validation_size: float = 0.2,
        elbow_sample_size: int = 20_000,
        random_state: int = 101,
    ) -> None:
        self.max_clusters = max_clusters
        self.c_values = c_values
        self.imbalance_cutoffs = imbalance_cutoffs
        self.validation_splits = validation_splits
        self.validation_size = validation_size
        self.elbow_sample_size = elbow_sample_size
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any):
        """Fit preprocessing, select parameters, and train local experts."""
        self._validate_parameters()
        frame = self._prepare_frame(X)
        target = self._validate_target(y, len(frame))

        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.classes_ = np.asarray([0, 1], dtype=int)
        self.preprocessor_ = self._new_preprocessor(frame)
        train_features = self._transform_fit(self.preprocessor_, frame)

        self.selected_k_, self.wcss_curve_ = self._select_k(train_features)
        self.selected_c_, self.selected_cutoff_, self.validation_results_ = (
            self._select_local_parameters(frame, target)
        )

        self.clusterer_ = KMeans(
            n_clusters=self.selected_k_,
            init="k-means++",
            n_init=50,
            random_state=self.random_state,
        ).fit(train_features)
        self.experts_ = self._fit_experts(
            train_features,
            target,
            self.clusterer_,
            self.selected_c_,
            self.selected_cutoff_,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probabilities in class order [0, 1]."""
        check_is_fitted(self, ["preprocessor_", "clusterer_", "experts_"])
        frame = self._prepare_frame(X)
        expected = list(self.feature_names_in_)
        if set(frame.columns) != set(expected):
            missing = sorted(set(expected) - set(frame.columns))
            extra = sorted(set(frame.columns) - set(expected))
            raise ValueError(f"Feature columns differ: missing={missing}, extra={extra}")
        values = np.asarray(self.preprocessor_.transform(frame[expected]), dtype=float)
        positive = self._routed_positive_probability(values, self.clusterer_, self.experts_)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary labels using the fixed probability threshold 0.50."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def cluster_report(self) -> pd.DataFrame:
        """Summarize pure, weighted, and standard local clusters."""
        check_is_fitted(self, ["experts_"])
        return pd.DataFrame(
            {
                "cluster": index + 1,
                "n_train": expert.n_samples,
                "n_positive": expert.n_positive,
                "minority_share": expert.minority_share,
                "fit_mode": expert.mode,
            }
            for index, expert in enumerate(self.experts_)
        )

    def _validate_parameters(self) -> None:
        if not isinstance(self.random_state, int) or self.random_state < 0:
            raise ValueError("random_state must be a nonnegative integer")
        if not isinstance(self.max_clusters, int) or self.max_clusters < 2:
            raise ValueError("max_clusters must be an integer >= 2")
        if not isinstance(self.validation_splits, int) or self.validation_splits < 1:
            raise ValueError("validation_splits must be an integer >= 1")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must lie strictly between 0 and 1")
        if not isinstance(self.elbow_sample_size, int) or self.elbow_sample_size < 3:
            raise ValueError("elbow_sample_size must be an integer >= 3")
        if not self.c_values or any(c <= 0 for c in self.c_values):
            raise ValueError("c_values must contain positive numbers")
        if not self.imbalance_cutoffs or any(
            not 0 < cutoff <= 0.5 for cutoff in self.imbalance_cutoffs
        ):
            raise ValueError("imbalance_cutoffs must lie in (0, 0.5]")

    @staticmethod
    def _prepare_frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if X.empty or X.shape[1] == 0:
            raise ValueError("X must have at least one row and one feature")
        if X.columns.has_duplicates or not all(isinstance(c, str) for c in X.columns):
            raise ValueError("Feature names must be unique strings")
        frame = X.copy()
        for column in frame.columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                series = pd.to_numeric(series, errors="raise").astype(float)
                if np.isinf(series.to_numpy(dtype=float)).any():
                    raise ValueError(f"Infinite numeric value in {column!r}")
                frame[column] = series
            else:
                frame[column] = series.map(
                    lambda value: "__missing__" if pd.isna(value) else str(value)
                )
        return frame

    @staticmethod
    def _validate_target(y: Any, n_rows: int) -> np.ndarray:
        values = np.asarray(y)
        if values.ndim != 1 or len(values) != n_rows:
            raise ValueError("y must contain one binary label for each row of X")
        if pd.isna(values).any() or set(np.unique(values).tolist()) != {0, 1}:
            raise ValueError("y must contain both classes encoded as 0 and 1")
        counts = np.bincount(values.astype(int), minlength=2)
        if counts.min() < 3:
            raise ValueError("At least three examples of each class are needed for validation")
        return values.astype(int)

    @staticmethod
    def _new_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
        numeric = [
            col for col in frame
            if pd.api.types.is_numeric_dtype(frame[col])
        ]
        categorical = [col for col in frame if col not in numeric]
        steps = []
        if numeric:
            steps.append(
                ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric)
            )
        if categorical:
            steps.append(
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical,
                )
            )
        return ColumnTransformer(steps, remainder="drop", sparse_threshold=0)

    @staticmethod
    def _transform_fit(transformer: ColumnTransformer, frame: pd.DataFrame) -> np.ndarray:
        values = np.asarray(transformer.fit_transform(frame), dtype=float)
        if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
            raise ValueError("Preprocessing produced invalid features")
        return values

    def _select_k(self, features: np.ndarray) -> tuple[int, pd.DataFrame]:
        rng = np.random.default_rng(self.random_state + 4242)
        sample_indices = rng.choice(
            len(features),
            size=min(len(features), self.elbow_sample_size),
            replace=False,
        )
        sample = features[sample_indices]
        n_unique = len(np.unique(sample, axis=0))
        if n_unique == 1:
            return 1, pd.DataFrame({"k": [1], "wcss": [0.0], "normalized_gap": [0.0]})
        last_candidate = min(self.max_clusters + 1, n_unique)
        candidates = np.arange(1, last_candidate + 1)
        wcss = np.asarray(
            [
                KMeans(
                    n_clusters=int(k),
                    init="k-means++",
                    n_init=10,
                    random_state=self.random_state + int(k),
                ).fit(sample).inertia_
                for k in candidates
            ],
            dtype=float,
        )
        # Stochastic optimization may make a later candidate slightly worse.
        wcss = np.minimum.accumulate(wcss)
        if last_candidate <= 2 or wcss[0] == wcss[-1]:
            gaps = np.zeros(len(candidates), dtype=float)
            selected = min(2, last_candidate)
        else:
            x = (candidates - 1) / (last_candidate - 1)
            y = (wcss - wcss[-1]) / (wcss[0] - wcss[-1])
            gaps = 1.0 - x - y
            # Excluding endpoints gives a selectable range 2..max_clusters.
            selected = int(candidates[1:-1][np.argmax(gaps[1:-1])])
        curve = pd.DataFrame(
            {"k": candidates, "wcss": wcss, "normalized_gap": gaps}
        )
        return selected, curve

    def _select_local_parameters(
        self, frame: pd.DataFrame, target: np.ndarray
    ) -> tuple[float, float, pd.DataFrame]:
        split = StratifiedShuffleSplit(
            n_splits=self.validation_splits,
            test_size=self.validation_size,
            random_state=self.random_state + 10_000,
        )
        records: list[dict[str, float | int]] = []
        try:
            folds = list(split.split(frame, target))
        except ValueError as exc:
            raise ValueError("Not enough rows per class for internal validation") from exc

        for fold_index, (train_idx, valid_idx) in enumerate(folds):
            inner_train = frame.iloc[train_idx]
            inner_valid = frame.iloc[valid_idx]
            transformer = self._new_preprocessor(inner_train)
            train_features = self._transform_fit(transformer, inner_train)
            valid_features = np.asarray(transformer.transform(inner_valid), dtype=float)
            distinct = len(np.unique(train_features, axis=0))
            if distinct < self.selected_k_:
                raise ValueError(
                    "An internal training split has fewer distinct feature rows "
                    "than the selected number of clusters"
                )
            gate = KMeans(
                n_clusters=self.selected_k_,
                init="k-means++",
                n_init=10,
                random_state=self.random_state * 10 + fold_index,
            ).fit(train_features)
            for c in self.c_values:
                for cutoff in self.imbalance_cutoffs:
                    experts = self._fit_experts(
                        train_features, target[train_idx], gate, float(c), float(cutoff)
                    )
                    probability = self._routed_positive_probability(
                        valid_features, gate, experts
                    )
                    records.append(
                        {
                            "fold": fold_index,
                            "C": float(c),
                            "imbalance_cutoff": float(cutoff),
                            "f1": float(
                                f1_score(
                                    target[valid_idx],
                                    probability >= 0.5,
                                    zero_division=0,
                                )
                            ),
                        }
                    )

        results = pd.DataFrame(records)
        ranked = (
            results.groupby(["C", "imbalance_cutoff"], as_index=False)["f1"]
            .mean()
            .sort_values(
                ["f1", "C", "imbalance_cutoff"],
                ascending=[False, True, True],
            )
            .reset_index(drop=True)
        )
        winner = ranked.iloc[0]
        return float(winner["C"]), float(winner["imbalance_cutoff"]), ranked

    @staticmethod
    def _fit_experts(
        features: np.ndarray,
        target: np.ndarray,
        gate: KMeans,
        c: float,
        cutoff: float,
    ) -> list[_ClusterExpert]:
        assigned = gate.predict(features)
        experts = []
        for cluster_index in range(gate.n_clusters):
            in_cluster = assigned == cluster_index
            y_local = target[in_cluster]
            n_positive = int(y_local.sum())
            n_negative = int(len(y_local) - n_positive)
            if len(y_local) == 0:
                raise RuntimeError("K-means++ produced an empty training cluster")
            minority_share = min(n_positive, n_negative) / len(y_local)
            if min(n_positive, n_negative) == 0:
                experts.append(
                    _ClusterExpert(
                        "pure", len(y_local), n_positive, minority_share,
                        int(y_local[0]), None,
                    )
                )
                continue
            weighted = minority_share < cutoff
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "logistic",
                        LogisticRegression(
                            C=c,
                            solver="lbfgs",
                            max_iter=1000,
                            class_weight="balanced" if weighted else None,
                        ),
                    ),
                ]
            )
            model.fit(features[in_cluster], y_local)
            experts.append(
                _ClusterExpert(
                    "weighted" if weighted else "standard",
                    len(y_local), n_positive, minority_share, None, model,
                )
            )
        return experts

    @staticmethod
    def _routed_positive_probability(
        features: np.ndarray, gate: KMeans, experts: list[_ClusterExpert]
    ) -> np.ndarray:
        assigned = gate.predict(features)
        probability = np.empty(len(features), dtype=float)
        for index, expert in enumerate(experts):
            in_cluster = assigned == index
            if not in_cluster.any():
                continue
            if expert.pure_label is not None:
                probability[in_cluster] = float(expert.pure_label)
            else:
                assert expert.model is not None
                probability[in_cluster] = expert.model.predict_proba(
                    features[in_cluster]
                )[:, 1]
        if not np.isfinite(probability).all():
            raise RuntimeError("A routed probability is missing or nonfinite")
        return probability
