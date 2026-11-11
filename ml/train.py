"""Train and evaluate XGBoost with chronological time-splits, typology holdouts, and calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from xgboost import XGBClassifier

from finvigil.config import PROJECT_ROOT, settings
from finvigil.db.models import Account
from finvigil.db.session import session_scope
from ml.features import FEATURE_NAMES, AccountFeatureVector, extract_account_features

SETTLEMENT_STUB_IDS = {
    "C8801",
    "C8802",
    "C8803",
    "C8804",
    "M9940",
    "M9955",
    "C1000",
    "E2001",
    "E2002",
    "E2003",
}


def _compute_detailed_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute Precision, Recall, False Positives, FPR, F1, PR-AUC, ROC-AUC, and Brier score."""
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp) / max(1.0, float(fp + tn))
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "false_positive_rate": round(fpr, 4),
        "brier_calibration_loss": round(float(brier_score_loss(y_true, y_probs)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_probs)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_probs)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_and_register_model(
    db_url: str | None = None,
    artifact_path: Path | None = None,
    *,
    log_to_mlflow: bool = False,
) -> dict[str, Any]:
    """Train LogisticRegression baseline and XGBoost on a strict chronological split + typology holdout."""
    target_artifact = artifact_path or settings.risk_model_path
    target_artifact.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = target_artifact.with_name("model_metadata.json")

    with session_scope(db_url, readonly=True) as session:
        account_ids = [
            aid
            for aid in session.scalars(select(Account.account_id)).all()
            if aid not in SETTLEMENT_STUB_IDS
        ]
        # Extract point-in-time features up to each account's decision timestamp
        vectors: list[AccountFeatureVector] = []
        for aid in account_ids:
            v = extract_account_features(session, aid)
            if v.features["tx_count"] > 0:
                vectors.append(extract_account_features(session, aid, as_of_step=v.max_step_hour))

    # ------------------------------------------------------------------
    # 1. Chronological Time-Split (Train <= 480h, Val 481..590h, Test > 590h)
    # ------------------------------------------------------------------
    train_vecs = [v for v in vectors if v.max_step_hour <= 480]
    val_vecs = [v for v in vectors if 480 < v.max_step_hour <= 590]
    test_vecs = [v for v in vectors if v.max_step_hour > 590]

    x_train = np.nan_to_num(np.vstack([v.to_numpy() for v in train_vecs]), nan=0.0)
    y_train = np.array([v.label for v in train_vecs], dtype=np.int32)
    x_val = np.nan_to_num(np.vstack([v.to_numpy() for v in val_vecs]), nan=0.0)
    y_val = np.array([v.label for v in val_vecs], dtype=np.int32)
    x_test = np.nan_to_num(np.vstack([v.to_numpy() for v in test_vecs]), nan=0.0)
    y_test = np.array([v.label for v in test_vecs], dtype=np.int32)

    # Fit StandardScaler strictly on chronological training partition
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    # Baseline: Logistic Regression
    baseline_clf = LogisticRegression(random_state=42, max_iter=500)
    baseline_clf.fit(x_train_scaled, y_train)
    val_probs_base = baseline_clf.predict_proba(x_val_scaled)[:, 1]
    test_probs_base = baseline_clf.predict_proba(x_test_scaled)[:, 1]
    baseline_metrics = {
        "val": _compute_detailed_metrics(y_val, val_probs_base, settings.freeze_risk_threshold),
        "test": _compute_detailed_metrics(y_test, test_probs_base, settings.freeze_risk_threshold),
    }

    # Production Model: Regularized XGBoost on chronological split
    xgb_clf = XGBClassifier(
        n_estimators=45,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_clf.fit(x_train, y_train)

    val_probs_xgb = xgb_clf.predict_proba(x_val)[:, 1]
    test_probs_xgb = xgb_clf.predict_proba(x_test)[:, 1]

    chronological_xgb_metrics: dict[str, Any] = {
        "val": _compute_detailed_metrics(y_val, val_probs_xgb, settings.freeze_risk_threshold),
        "test": _compute_detailed_metrics(y_test, test_probs_xgb, settings.freeze_risk_threshold),
    }

    # ------------------------------------------------------------------
    # 2. Unseen Typology Holdout Evaluation (Hold out LAYERING_MULE from train)
    # ------------------------------------------------------------------
    typo_train_vecs = [
        v
        for v in vectors
        if v.typology != "LAYERING_MULE" and v.max_step_hour <= 540
    ]
    typo_holdout_vecs = [
        v
        for v in vectors
        if v.typology == "LAYERING_MULE" or v.max_step_hour > 540
    ]
    x_typo_train = np.nan_to_num(np.vstack([v.to_numpy() for v in typo_train_vecs]), nan=0.0)
    y_typo_train = np.array([v.label for v in typo_train_vecs], dtype=np.int32)
    x_typo_test = np.nan_to_num(np.vstack([v.to_numpy() for v in typo_holdout_vecs]), nan=0.0)
    y_typo_test = np.array([v.label for v in typo_holdout_vecs], dtype=np.int32)

    typo_clf = XGBClassifier(
        n_estimators=45,
        max_depth=3,
        learning_rate=0.08,
        reg_lambda=2.0,
        eval_metric="logloss",
        random_state=42,
    )
    typo_clf.fit(x_typo_train, y_typo_train)
    typo_probs = typo_clf.predict_proba(x_typo_test)[:, 1]
    typology_holdout_metrics = _compute_detailed_metrics(
        y_typo_test, typo_probs, settings.freeze_risk_threshold
    )

    xgb_clf.save_model(str(target_artifact))

    metadata: dict[str, Any] = {
        "model_version": settings.risk_model_version,
        "feature_names": list(FEATURE_NAMES),
        "freeze_threshold": settings.freeze_risk_threshold,
        "split_strategy": "chronological_step_hour_and_typology_holdout",
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "baseline_logistic_regression": baseline_metrics,
        "xgboost_chronological_metrics": chronological_xgb_metrics,
        "xgboost_typology_holdout_metrics": typology_holdout_metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if log_to_mlflow:
        try:
            import mlflow

            mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
            mlflow.set_experiment("finvigil-aml-xgboost")
            with mlflow.start_run(run_name=settings.risk_model_version):
                mlflow.log_params(
                    {
                        "model_version": settings.risk_model_version,
                        "n_estimators": 45,
                        "max_depth": 3,
                        "learning_rate": 0.08,
                        "split": "chronological_and_typology",
                    }
                )
                test_m = chronological_xgb_metrics["test"]
                mlflow.log_metrics(
                    {
                        "test_precision": float(test_m["precision"]),
                        "test_recall": float(test_m["recall"]),
                        "test_f1": float(test_m["f1"]),
                        "test_fpr": float(test_m["false_positive_rate"]),
                        "test_brier_loss": float(test_m["brier_calibration_loss"]),
                        "test_pr_auc": float(test_m["pr_auc"]),
                        "typology_holdout_f1": float(typology_holdout_metrics["f1"]),
                    }
                )
                mlflow.log_artifact(str(target_artifact))
                mlflow.log_artifact(str(metadata_path))
        except Exception as exc:  # pragma: no cover
            metadata["mlflow_warning"] = str(exc)

    return metadata


if __name__ == "__main__":
    summary = train_and_register_model(log_to_mlflow=True)
    print(json.dumps(summary, indent=2))
