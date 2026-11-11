"""Deterministic XGBoost fraud/AML risk model client with point-in-time as_of_step cutoff."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ml.features import extract_account_features
from ml.train import train_and_register_model
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

from finvigil.config import settings
from finvigil.db.session import session_scope

_cached_model: XGBClassifier | None = None
_cached_path: Path | None = None


class RiskScoreResult(BaseModel):
    """Deterministic output from the versioned XGBoost risk model."""

    account_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_band: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    exceeds_freeze_threshold: bool
    freeze_threshold: float
    model_version: str
    as_of_step: int
    features: dict[str, float]
    top_risk_drivers: list[str]
    cited_transaction_ids: list[str]


def _load_xgb_model(artifact_path: Path | None = None) -> XGBClassifier:
    """Load or lazy-train the versioned XGBoost model artifact."""
    global _cached_model, _cached_path
    target = artifact_path or settings.risk_model_path
    if _cached_model is not None and _cached_path == target and target.exists():
        return _cached_model

    if not target.exists():
        train_and_register_model(artifact_path=target, log_to_mlflow=False)

    clf = XGBClassifier()
    clf.load_model(str(target))
    _cached_model = clf
    _cached_path = target
    return clf


def score_account_risk(
    account_id: str,
    *,
    as_of_step: int | None = None,
    db_url: str | None = None,
    artifact_path: Path | None = None,
) -> RiskScoreResult:
    """Compute point-in-time features from the ledger and evaluate the versioned XGBoost model."""
    clf = _load_xgb_model(artifact_path)

    with session_scope(db_url, readonly=True) as session:
        feat_vec = extract_account_features(session, account_id, as_of_step=as_of_step)

    x_input = feat_vec.to_numpy().reshape(1, -1)
    prob = float(clf.predict_proba(x_input)[0, 1])
    score = round(prob, 4)

    if score >= 0.85:
        band: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "CRITICAL"
    elif score >= settings.freeze_risk_threshold:
        band = "HIGH"
    elif score >= 0.35:
        band = "MEDIUM"
    else:
        band = "LOW"

    feats = feat_vec.features
    drivers: list[str] = []
    if feats["structuring_tx_count"] >= 2:
        drivers.append(
            f"Structuring/smurfing pattern: {int(feats['structuring_tx_count'])} sub-10k EUR transfers"
        )
    if feats["max_single_drain_ratio"] >= 0.70:
        drivers.append(
            f"Rapid balance depletion: max single outbound drain ratio {feats['max_single_drain_ratio']:.1%}"
        )
    if feats["high_risk_corridor_ratio"] >= 0.40:
        drivers.append(
            f"High-risk counterparty jurisdiction exposure ({feats['high_risk_corridor_ratio']:.1%} of txs)"
        )
    if feats["remaining_balance_ratio"] <= 0.20:
        drivers.append(
            f"Near-zero balance retention ({feats['remaining_balance_ratio']:.1%} remaining)"
        )
    if feats["kyc_tier"] <= 1.0 and feats["account_age_days"] < 150:
        drivers.append(
            f"Low KYC tier ({int(feats['kyc_tier'])}) on young account ({int(feats['account_age_days'])} days)"
        )
    if not drivers:
        drivers.append(
            f"Benign reserve retention ({feats['remaining_balance_ratio']:.1%}) and domestic payment profile"
        )

    return RiskScoreResult(
        account_id=account_id,
        risk_score=score,
        risk_band=band,
        exceeds_freeze_threshold=score >= settings.freeze_risk_threshold,
        freeze_threshold=settings.freeze_risk_threshold,
        model_version=settings.risk_model_version,
        as_of_step=feat_vec.max_step_hour,
        features=feats,
        top_risk_drivers=drivers,
        cited_transaction_ids=feat_vec.cited_tx_ids,
    )
