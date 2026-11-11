"""OpenTelemetry GenAI tracing and Prometheus metrics for FinVigil."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import Counter, Histogram

from finvigil.config import settings

_resource = Resource.create({"service.name": settings.service_name})
_provider = TracerProvider(resource=_resource)
SPAN_EXPORTER = InMemorySpanExporter()
_provider.add_span_processor(SimpleSpanProcessor(SPAN_EXPORTER))
trace.set_tracer_provider(_provider)

TRACER = trace.get_tracer("finvigil.genai", "0.1.0")

# Prometheus Metrics
INVESTIGATIONS_TOTAL = Counter(
    "finvigil_investigations_total",
    "Total AML/fraud investigations started",
    ["recommended_action", "status"],
)
AGENT_STEPS_TOTAL = Counter(
    "finvigil_agent_steps_total",
    "Total sub-agent steps executed across investigations",
    ["agent_name"],
)
TOKENS_USED_TOTAL = Counter(
    "finvigil_tokens_used_total",
    "Total LLM tokens consumed",
    ["direction"],
)
COST_EUR_TOTAL = Counter(
    "finvigil_cost_eur_total",
    "Cumulative LLM and tool execution cost in Euros",
)
SQL_GUARD_REJECTIONS_TOTAL = Counter(
    "finvigil_sql_guard_rejections_total",
    "Total SQL queries blocked by the sqlglot guard",
)
INVESTIGATION_COST_HISTOGRAM = Histogram(
    "finvigil_investigation_cost_eur",
    "Cost in EUR per investigation",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.10, 0.15, 0.25),
)


@contextmanager
def trace_agent_step(
    agent_name: str,
    *,
    case_id: str,
    account_id: str,
    step_number: int,
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span, None, None]:
    """Record an OpenTelemetry span for an agent step and increment Prometheus counters."""
    AGENT_STEPS_TOTAL.labels(agent_name=agent_name).inc()
    with TRACER.start_as_current_span(f"agent.{agent_name}") as span:
        span.set_attribute("gen_ai.system", settings.llm_provider)
        span.set_attribute("gen_ai.agent.name", agent_name)
        span.set_attribute("finvigil.case_id", case_id)
        span.set_attribute("finvigil.account_id", account_id)
        span.set_attribute("finvigil.step_number", step_number)
        if attributes:
            for k, v in attributes.items():
                if isinstance(v, (str, int, float, bool)):
                    span.set_attribute(k, v)
        yield span
