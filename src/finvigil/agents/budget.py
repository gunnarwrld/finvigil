"""Hard budget tracker enforcing caps on agent steps, tokens, and Euros per request."""

from __future__ import annotations

from pydantic import BaseModel

from finvigil.config import settings
from finvigil.observability import COST_EUR_TOTAL, TOKENS_USED_TOTAL


class BudgetSnapshot(BaseModel):
    """Per-request budget state tracked by the Supervisor agent."""

    steps_used: int = 0
    max_steps: int = settings.max_steps_per_request
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_used: int = 0
    max_tokens: int = settings.max_tokens_per_request
    cost_eur: float = 0.0
    max_cost_eur: float = settings.max_cost_eur_per_request
    is_exhausted: bool = False
    exhausted_reason: str | None = None

    def record_step(self, *, in_tokens: int, out_tokens: int) -> BudgetSnapshot:
        """Return an updated BudgetSnapshot after accounting for one agent step."""
        new_steps = self.steps_used + 1
        new_in = self.input_tokens + in_tokens
        new_out = self.output_tokens + out_tokens
        new_total_tokens = new_in + new_out

        step_cost = (
            (in_tokens / 1000.0) * settings.input_cost_eur_per_1k_tokens
            + (out_tokens / 1000.0) * settings.output_cost_eur_per_1k_tokens
        )
        new_cost = round(self.cost_eur + step_cost, 6)

        TOKENS_USED_TOTAL.labels(direction="input").inc(in_tokens)
        TOKENS_USED_TOTAL.labels(direction="output").inc(out_tokens)
        COST_EUR_TOTAL.inc(step_cost)

        reason: str | None = self.exhausted_reason
        if new_steps >= self.max_steps:
            reason = f"Step budget reached ({new_steps}/{self.max_steps} steps)"
        elif new_total_tokens >= self.max_tokens:
            reason = f"Token budget reached ({new_total_tokens}/{self.max_tokens} tokens)"
        elif new_cost >= self.max_cost_eur:
            reason = f"Euro budget reached (€{new_cost:.4f}/€{self.max_cost_eur:.4f})"

        return BudgetSnapshot(
            steps_used=new_steps,
            max_steps=self.max_steps,
            input_tokens=new_in,
            output_tokens=new_out,
            tokens_used=new_total_tokens,
            max_tokens=self.max_tokens,
            cost_eur=new_cost,
            max_cost_eur=self.max_cost_eur,
            is_exhausted=reason is not None,
            exhausted_reason=reason,
        )
