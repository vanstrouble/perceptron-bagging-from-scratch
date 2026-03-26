from __future__ import annotations

from typing import Any, Callable

import numpy as np


class BaggingClassifier:
    """Model-agnostic Bagging classifier for estimators exposing fit/predict."""

    def __init__(
        self,
        estimator_factory: Callable[[], Any],
        n_estimators: int = 10,
        sample_ratio: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        """
        :param estimator_factory: Zero-arg callable returning a new estimator instance.
            The estimator must implement fit(X, y) and predict(X).
        :param n_estimators: Number of bootstrap estimators to train.
        :param sample_ratio: Fraction of training data used per bootstrap sample.
        :param random_state: Seed for reproducible bootstrap sampling.
        """
        if not callable(estimator_factory):
            raise ValueError("estimator_factory must be callable.")
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0.")
        if sample_ratio <= 0:
            raise ValueError("sample_ratio must be greater than 0.")

        self.estimator_factory = estimator_factory
        self.n_estimators = n_estimators
        self.sample_ratio = sample_ratio
        self.random_state = random_state

        self.estimators_: list[Any] = []
        self.classes_: np.ndarray | None = None

    def _bootstrap_indices(self, n_samples: int, rng: np.random.RandomState) -> np.ndarray:
        size = max(1, int(round(self.sample_ratio * n_samples)))
        return rng.randint(0, n_samples, size=size)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingClassifier":
        """Train n_estimators models on bootstrap samples."""
        if len(X) != len(y):
            raise ValueError("The number of samples and labels must be the same.")
        if len(X) == 0:
            raise ValueError("Training data cannot be empty.")

        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("At least two classes are required for classification.")

        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []

        for _ in range(self.n_estimators):
            idx = self._bootstrap_indices(len(X), rng)
            X_boot = X[idx]
            y_boot = y[idx]

            estimator = self.estimator_factory()
            if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
                raise TypeError("Estimator must expose fit(X, y) and predict(X).")

            estimator.fit(X_boot, y_boot)
            self.estimators_.append(estimator)

        return self

    def _collect_predictions(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise ValueError("The ensemble has not been trained yet.")

        predictions = [estimator.predict(X) for estimator in self.estimators_]
        return np.asarray(predictions)

    def _majority_vote(self, votes: np.ndarray) -> Any:
        values, counts = np.unique(votes, return_counts=True)
        winners = values[counts == np.max(counts)]

        if len(winners) == 1:
            return winners[0]

        # Deterministic tie break: use the original class order from fit().
        for cls in self.classes_:
            if np.any(winners == cls):
                return cls

        return winners[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels using majority vote across all estimators."""
        if self.classes_ is None:
            raise ValueError("The ensemble has not been trained yet.")

        all_preds = self._collect_predictions(X)
        final_preds = [self._majority_vote(sample_votes) for sample_votes in all_preds.T]
        return np.asarray(final_preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate class probabilities as vote frequencies."""
        if self.classes_ is None:
            raise ValueError("The ensemble has not been trained yet.")

        all_preds = self._collect_predictions(X)
        n_samples = all_preds.shape[1]
        probabilities = np.zeros((n_samples, len(self.classes_)), dtype=float)

        for i, cls in enumerate(self.classes_):
            probabilities[:, i] = np.mean(all_preds == cls, axis=0)

        return probabilities

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score."""
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
