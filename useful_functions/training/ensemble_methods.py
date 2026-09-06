"""Ensemble model comparison utilities: train and evaluate Bagging, Boosting, and Stacking classifiers."""

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report


def ensembles(X: pd.DataFrame, y: np.ndarray, random_state: int = 42) -> dict[str, dict[str, float]]:
    """Train Bagging, Boosting, and Stacking classifiers and compare their test-set metrics.

    Args:
        X: Feature DataFrame (already encoded/scaled — see `encoding()` and
            `dataloader()`).
        y: Target array (class labels).
        random_state: Seed passed to every model and to the train/test
            split (via `dataloader`) for reproducible comparisons across
            runs. Without a fixed seed, apparent performance
            differences between models could just be split/initialization
            noise rather than real differences.

    Returns:
        A dict keyed by model name ("bagging", "boosting", "stacking"),
        each mapping to a dict of weighted-average metrics:
        {"accuracy": float, "f1": float, "precision": float, "recall": float}.
        Weighted averages account for class imbalance by weighting
        each class's contribution by its support (true sample count).

    Example:
        >>> results = ensembles(X, y, random_state=0)
        >>> results["stacking"]["f1"]
        0.87
    """
    X_train, X_test, y_train, y_test = dataloader(X, y, random_state=random_state)

    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=random_state),
        n_estimators=100,
        max_samples=0.8,
        random_state=random_state,
    )
    boosting = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=random_state),
        n_estimators=100,
        random_state=random_state,
    )
    stacking = StackingClassifier(
        estimators=[
            ("lr", LogisticRegression()),
            ("svc", SVC(probability=True)),
            ("tree", DecisionTreeClassifier(random_state=random_state)),
        ],
        final_estimator=LogisticRegression(),
    )

    models = {"bagging": bagging, "boosting": boosting, "stacking": stacking}
    results: dict[str, dict[str, float]] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # y_true first, y_pred second — reversing these mislabels ground truth
        report = classification_report(y_test, y_pred, output_dict=True)
        weighted = report["weighted avg"]  # accounts for **class imbalance**

        results[model_name] = {
            "accuracy": report["accuracy"],
            "f1": weighted["f1-score"],
            "precision": weighted["precision"],
            "recall": weighted["recall"],
        }

    return results
