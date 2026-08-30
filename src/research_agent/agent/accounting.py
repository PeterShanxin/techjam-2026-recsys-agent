"""Resource ledger. Token counts are source of truth. Dollar cost is optional metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research_agent.llm.types import UsageRecord


@dataclass
class ResourceLedger:
    llm_calls: int = 0
    research_calls: int = 0
    repair_calls: int = 0
    smoke_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    llm_latency_seconds: float = 0.0
    experiment_runtime_seconds: float = 0.0
    research_wall_seconds: float = 0.0
    completed_experiments: int = 0
    failed_experiments: int = 0
    manual_interventions: int = 0
    transport_retries: int = 0
    calls: list[UsageRecord] = field(default_factory=list)

    def add_usage(self, usage: UsageRecord) -> None:
        self.llm_calls += 1
        if usage.purpose == "research":
            self.research_calls += 1
        elif usage.purpose == "repair":
            self.repair_calls += 1
        elif usage.purpose == "smoke":
            self.smoke_calls += 1
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0
        self.thinking_tokens += usage.thinking_tokens or 0
        self.cached_tokens += usage.cached_tokens or 0
        self.total_tokens += usage.total_tokens or 0
        self.llm_latency_seconds += float(usage.latency_seconds)
        self.calls.append(usage)

    def add_experiment(self, *, status: str, wall_seconds: float) -> None:
        self.experiment_runtime_seconds += float(wall_seconds)
        if status == "success":
            self.completed_experiments += 1
        else:
            self.failed_experiments += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "research_calls": self.research_calls,
            "repair_calls": self.repair_calls,
            "smoke_calls": self.smoke_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "llm_latency_seconds": float(self.llm_latency_seconds),
            "experiment_runtime_seconds": float(self.experiment_runtime_seconds),
            "research_wall_seconds": float(self.research_wall_seconds),
            "completed_experiments": self.completed_experiments,
            "failed_experiments": self.failed_experiments,
            "manual_interventions": self.manual_interventions,
            "transport_retries": self.transport_retries,
        }
