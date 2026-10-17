"""Traditional ML classifiers on statistical flow features (review Section 3.2).

Random Forest is the representative of the "machine learning" generation; XGBoost is an
optional extra baseline. Model size is reported as the total number of tree nodes, the
closest analogue of a parameter count for tree ensembles.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier


def build_rf(params: dict[str, Any], seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=params.get("max_depth"),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        class_weight=params.get("class_weight", "balanced"),
        n_jobs=-1,
        random_state=seed,
    )


def build_xgb(params: dict[str, Any], seed: int, num_classes: int) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as e:  # optional dependency
        raise ImportError("XGBoost is optional: run `make setup-xgb` (macOS: brew install libomp)") from e
    return XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        objective="multi:softprob",
        num_class=num_classes,
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )


def tree_node_count(model: Any) -> int:
    """Total number of nodes over all trees (RF or XGBoost)."""
    if hasattr(model, "estimators_"):
        return int(sum(est.tree_.node_count for est in model.estimators_))
    booster = model.get_booster()
    return int(len(booster.trees_to_dataframe()))
