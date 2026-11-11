"""Point-in-time account feature engineering from double-entry integer-cent transactions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from finvigil.db.models import Account, Transaction

HIGH_RISK_COUNTRIES = {"CY", "MT", "EE", "LV", "LT"}

FEATURE_NAMES: tuple[str, ...] = (
    "kyc_tier",
    "account_age_days",
    "tx_count",
    "structuring_tx_count",
    "cash_out_or_transfer_ratio",
    "max_single_drain_ratio",
    "high_risk_corridor_ratio",
    "outbound_to_inbound_ratio",
    "remaining_balance_ratio",
)


@dataclass(frozen=True)
class AccountFeatureVector:
    """Point-in-time quantitative features, cited transactions, typology, and latest step."""

    account_id: str
    features: dict[str, float]
    cited_tx_ids: list[str]
    label: int
    typology: str
    max_step_hour: int

    def to_numpy(self) -> np.ndarray:
        """Convert feature dictionary into ordered 1D float32 numpy array."""
        return np.array([self.features[name] for name in FEATURE_NAMES], dtype=np.float32)


def extract_account_features(
    session: Session,
    account_id: str,
    *,
    as_of_step: int | None = None,
) -> AccountFeatureVector:
    """Compute point-in-time PaySim + AML features strictly up to `as_of_step` (no future leakage)."""
    acct = session.get(Account, account_id)
    if acct is None:
        raise ValueError(f"Account '{account_id}' not found in ledger.")

    stmt = select(Transaction).where(
        or_(
            Transaction.sender_account_id == account_id,
            Transaction.receiver_account_id == account_id,
        )
    )
    if as_of_step is not None:
        stmt = stmt.where(Transaction.step_hour <= as_of_step)

    stmt = stmt.order_by(Transaction.step_hour.asc(), Transaction.tx_id.asc())
    txs = list(session.scalars(stmt).all())

    if not txs:
        zero_features = {
            "kyc_tier": float(acct.kyc_tier or 2),
            "account_age_days": float(acct.account_age_days or 365),
            "tx_count": 0.0,
            "structuring_tx_count": 0.0,
            "cash_out_or_transfer_ratio": 0.0,
            "max_single_drain_ratio": 0.0,
            "high_risk_corridor_ratio": 0.0,
            "outbound_to_inbound_ratio": 0.0,
            "remaining_balance_ratio": 1.0,
        }
        return AccountFeatureVector(
            account_id=account_id,
            features=zero_features,
            cited_tx_ids=[],
            label=0,
            typology="BENIGN",
            max_step_hour=0,
        )

    inbound_txs = [t for t in txs if t.receiver_account_id == account_id]
    outbound_txs = [t for t in txs if t.sender_account_id == account_id]

    total_inbound_cents = sum(t.amount_cents for t in inbound_txs)
    total_outbound_cents = sum(t.amount_cents for t in outbound_txs)

    # Point-in-time balance as of the last observed transaction in the window
    last_tx = txs[-1]
    current_balance_cents = (
        last_tx.sender_new_balance_cents
        if last_tx.sender_account_id == account_id
        else last_tx.receiver_new_balance_cents
    )

    # Structuring detector: transfers between 850,000 and 999,999 cents (EUR 8,500..9,999.99)
    structuring_count = sum(
        1
        for t in txs
        if t.tx_type == "TRANSFER" and 850_000 <= t.amount_cents < 1_000_000
    )

    cash_out_or_transfer_count = sum(
        1 for t in outbound_txs if t.tx_type in {"CASH_OUT", "TRANSFER"}
    )
    cash_out_ratio = (
        cash_out_or_transfer_count / len(outbound_txs) if outbound_txs else 0.0
    )

    drain_ratios = [
        min(1.0, t.amount_cents / max(100, t.sender_old_balance_cents))
        for t in outbound_txs
    ]
    max_drain_ratio = max(drain_ratios) if drain_ratios else 0.0

    high_risk_count = sum(
        1 for t in txs if (t.counterparty_country or "SE") in HIGH_RISK_COUNTRIES
    )
    high_risk_ratio = high_risk_count / len(txs)

    outbound_to_inbound = total_outbound_cents / max(
        100, total_inbound_cents + current_balance_cents
    )
    peak_balance_cents = max(
        [current_balance_cents]
        + [t.sender_old_balance_cents for t in outbound_txs]
        + [t.receiver_new_balance_cents for t in inbound_txs]
    )
    remaining_ratio = min(1.0, current_balance_cents / max(100, peak_balance_cents))

    label = 1 if any(t.is_fraud_label == 1 for t in txs) else 0
    dominant_typology = last_tx.typology or ("FRAUD" if label == 1 else "BENIGN")
    max_step = max(t.step_hour for t in txs)

    feature_dict = {
        "kyc_tier": float(acct.kyc_tier if acct.kyc_tier is not None else 2),
        "account_age_days": float(
            acct.account_age_days if acct.account_age_days is not None else 365
        ),
        "tx_count": float(len(txs)),
        "structuring_tx_count": float(structuring_count),
        "cash_out_or_transfer_ratio": round(float(cash_out_ratio), 4),
        "max_single_drain_ratio": round(float(max_drain_ratio), 4),
        "high_risk_corridor_ratio": round(float(high_risk_ratio), 4),
        "outbound_to_inbound_ratio": round(float(outbound_to_inbound), 4),
        "remaining_balance_ratio": round(float(remaining_ratio), 4),
    }

    return AccountFeatureVector(
        account_id=account_id,
        features=feature_dict,
        cited_tx_ids=[t.tx_id for t in txs],
        label=label,
        typology=dominant_typology,
        max_step_hour=max_step,
    )
