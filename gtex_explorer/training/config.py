"""Bounded training controls, with no automatic hyperparameter search."""

import math
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from gtex_explorer.training.data import stable_hash

XGB_DEFAULTS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
    "n_jobs": 4,
}
# Ways to handle empty values in the training data: fill each gene's gaps with its median, or leave out
# every gene that has a gap.
MISSING_VALUES = ("median", "drop")
XGB_FIXED = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": 42,
    "verbosity": 0,
}


@dataclass(frozen=True)
class TrainingConfig:
    """A validated set of training choices."""

    selector: str
    top_k: int
    l1_c: float
    l1_max_iter: int
    l1_tol: float
    l1_class_weight: str | None
    mi_neighbors: int
    evaluate_cv: bool
    cv_folds: int
    xgboost_parameters: Mapping
    holdout_percent: int | None = None
    log2_transform: bool = True
    standardize: bool = True
    # How empty values in the training data are handled, and whether the technical filter runs.
    missing_values: str = "median"
    technical_filter: bool = True

    def preprocessing(self) -> dict:
        """The pra-pemrosesan choices that differ from the default (median, technical filter, log2(TPM+1), z-score).

        Empty for the default, so runs saved before these were choices keep their saved settings and identity.
        """
        result = {key: False for key in ("log2_transform", "standardize", "technical_filter") if getattr(self, key) is False}
        if self.missing_values != "median":
            result["missing_values"] = self.missing_values
        return result

    def transforms(self) -> dict:
        """The keyword arguments that prepare a partition with these choices."""
        return {
            "log2": self.log2_transform,
            "standardize": self.standardize,
            "missing": self.missing_values,
            "technical_filter": self.technical_filter,
        }

    def to_dict(self) -> dict:
        """Return the settings in the form saved with a run."""
        # Runs saved before hold-outs or preprocessing choices existed have no such entries, and their saved
        # config must still match, so these are written only when they differ from the old fixed behaviour.
        holdout = {"holdout_percent": self.holdout_percent} if self.holdout_percent else {}
        return {
            "selector": self.selector,
            "top_k": self.top_k,
            "l1_c": self.l1_c,
            "l1_max_iter": self.l1_max_iter,
            "l1_tol": self.l1_tol,
            "l1_class_weight": self.l1_class_weight,
            "mi_neighbors": self.mi_neighbors,
            "evaluate_cv": self.evaluate_cv,
            "cv_folds": self.cv_folds,
            "xgboost": {k: v for k, v in self.xgboost_parameters.items() if k not in XGB_FIXED},
            **holdout,
            **self.preprocessing(),
        }

    def selector_settings(self) -> dict:
        """Return the settings passed to the feature selector."""
        result = {"selector": self.selector}
        if self.selector == "l1":
            result.update(
                C=self.l1_c,
                max_iter=self.l1_max_iter,
                tol=self.l1_tol,
                class_weight=self.l1_class_weight,
                zero_tolerance=1e-12,
                solver="saga",
                random_state=42,
            )
        if self.selector == "mutual_information":
            result.update(n_neighbors=self.mi_neighbors, random_state=42)
        return result


def _number(value, low, high, name, integer=False):
    """Check that a setting is a finite number within its range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value) if integer else float(value)


def validate_training_config(
    values: dict, *, candidate_count: int = 5000, cpu_count: int | None = None
) -> TrainingConfig:
    """Check the training form values and build the configuration."""
    allowed = {
        "selector",
        "top_k",
        "l1_c",
        "l1_max_iter",
        "l1_tol",
        "l1_class_weight",
        "mi_neighbors",
        "evaluate_cv",
        "cv_folds",
        "holdout_percent",
        "log2_transform",
        "standardize",
        "missing_values",
        "technical_filter",
        "xgboost",
    }
    if set(values) - allowed:
        raise ValueError("Unknown training controls")
    selector = values.get("selector", "l1")
    if selector not in {"baseline", "anova", "mutual_information", "l1"}:
        raise ValueError("Unknown selector")
    k = _number(values.get("top_k", min(1000, candidate_count)), 1, min(5000, candidate_count), "top_k", True)
    c = _number(values.get("l1_c", 1.0), 0.0001, 1000, "C")
    iterations = _number(values.get("l1_max_iter", 20000), 100, 50000, "max_iter", True)
    tolerance = _number(values.get("l1_tol", 0.001), 0.000001, 0.01, "tol")
    weight = values.get("l1_class_weight", "balanced")
    if weight not in {"balanced", None}:
        raise ValueError("class_weight must be balanced or none")
    neighbors = _number(values.get("mi_neighbors", 3), 1, 20, "n_neighbors", True)
    folds = _number(values.get("cv_folds", 5), 2, 10, "folds", True)
    evaluate = values.get("evaluate_cv", False)
    if not isinstance(evaluate, bool):
        raise ValueError("evaluate_cv must be boolean")
    holdout = values.get("holdout_percent")
    if holdout is not None:
        holdout = _number(holdout, 5, 50, "holdout_percent", True)
    log2 = values.get("log2_transform", True)
    standardize = values.get("standardize", True)
    if not isinstance(log2, bool) or not isinstance(standardize, bool):
        raise ValueError("log2_transform and standardize must be boolean")
    missing = values.get("missing_values", "median")
    if missing not in MISSING_VALUES:
        raise ValueError("missing_values must be median or drop")
    technical_filter = values.get("technical_filter", True)
    if not isinstance(technical_filter, bool):
        raise ValueError("technical_filter must be boolean")
    custom = values.get("xgboost", {})
    if not isinstance(custom, dict) or set(custom) - set(XGB_DEFAULTS):
        raise ValueError("Unknown or fixed XGBoost parameter")
    parameters = {**XGB_DEFAULTS, **custom}
    limits = {
        "n_estimators": (1, 1000, True),
        "learning_rate": (0.001, 1, False),
        "max_depth": (1, 12, True),
        "min_child_weight": (0, 100, False),
        "subsample": (0.1, 1, False),
        "colsample_bytree": (0.1, 1, False),
        "reg_lambda": (0, 10000, False),
        "reg_alpha": (0, 10000, False),
        "gamma": (0, 1000, False),
        "n_jobs": (1, max(1, cpu_count or os.cpu_count() or 1), True),
    }
    # Default threads are capped, but an explicit out-of-bounds request is rejected.
    if "n_jobs" not in custom:
        parameters["n_jobs"] = min(4, limits["n_jobs"][1])
    for key, (low, high, integer) in limits.items():
        parameters[key] = _number(parameters[key], low, high, key, integer)
    parameters.update(XGB_FIXED)
    return TrainingConfig(
        selector,
        candidate_count if selector == "baseline" else k,
        c,
        iterations,
        tolerance,
        weight,
        neighbors,
        evaluate,
        folds,
        MappingProxyType(parameters),
        holdout,
        log2,
        standardize,
        missing,
        technical_filter,
    )


def configuration_fingerprint(
    config: TrainingConfig, data_fingerprint: str, split_hash: str | None, versions: dict
) -> str:
    """Return a hash that identifies this training setup."""
    holdout = {"holdout": config.holdout_percent} if config.holdout_percent else {}
    # Left out for the default transforms, so runs saved before this choice existed keep their identity.
    preprocessing = {"preprocessing": config.preprocessing()} if config.preprocessing() else {}
    return stable_hash(
        {
            "selector": config.selector_settings(),
            "k": config.top_k,
            "xgboost": dict(config.xgboost_parameters),
            "data": data_fingerprint,
            "split": split_hash if config.evaluate_cv or config.holdout_percent else None,
            "folds": config.cv_folds if config.evaluate_cv else None,
            "versions": versions,
            **holdout,
            **preprocessing,
        }
    )
