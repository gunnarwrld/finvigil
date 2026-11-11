"""Authentication and case-binding verification for Human-in-the-Loop freeze approval."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from fastapi import Header, HTTPException

from finvigil.config import settings


def compute_case_file_hash(case_file: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 digest binding an approval to the exact CaseFile reviewed."""
    canonical_fields = {
        "case_id": case_file.get("case_id"),
        "account_id": case_file.get("account_id"),
        "risk_score": case_file.get("risk_score"),
        "model_version": case_file.get("model_version"),
        "cited_transaction_ids": sorted(str(x) for x in case_file.get("cited_transaction_ids", [])),
        "recommended_action": case_file.get("recommended_action"),
    }
    serialized = json.dumps(canonical_fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def verify_compliance_officer_token(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> str:
    """Authenticate Bearer token and return the verified compliance officer identity."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or malformed Authorization Bearer token.",
        )

    supplied_token = authorization.removeprefix("Bearer ").strip()
    if not supplied_token:
        raise HTTPException(status_code=401, detail="Empty Bearer token.")

    try:
        token_map: dict[str, str] = json.loads(settings.auth_tokens_json)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Invalid auth configuration.") from exc

    for configured_token, officer_id in token_map.items():
        if hmac.compare_digest(supplied_token.encode("utf-8"), configured_token.encode("utf-8")):
            return officer_id

    raise HTTPException(
        status_code=401,
        detail="Invalid compliance officer Bearer token.",
    )
