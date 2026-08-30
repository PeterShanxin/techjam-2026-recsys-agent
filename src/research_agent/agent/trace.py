"""Research trace persistence and human-readable export."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from research_agent.llm.secrets import assert_no_secrets, sanitize


@dataclass
class ResearchTrace:
    path: Path
    report_path: Path
    summary_path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def append(self, record: dict[str, Any]) -> None:
        payload = sanitize(record)
        assert_no_secrets(payload)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def write_exports(self, *, summary: dict[str, Any], records: Iterable[dict[str, Any]] | None = None) -> None:
        rows = list(records) if records is not None else self.records()
        payload = sanitize(summary)
        assert_no_secrets(payload)
        self.summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        self.report_path.write_text(render_markdown(payload, rows), encoding="utf-8")


def render_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 3 research trace",
        "",
        f"- model: `{summary.get('model')}`",
        f"- thinking: `{summary.get('thinking_level')}`",
        f"- manual interventions: {summary.get('manual_interventions', 0)}",
        f"- research wall-clock (s): {summary.get('research_wall_seconds')}",
        "",
        "## Resource usage",
        "",
        "```json",
        json.dumps(summary.get("resources", {}), indent=2, ensure_ascii=True),
        "```",
        "",
        "## Iterations",
        "",
    ]
    for row in records:
        metrics = row.get("metrics") or {}
        lines.extend(
            [
                f"### {row.get('iteration')} `{row.get('experiment_id')}`",
                "",
                f"- parent: `{row.get('parent_id')}`",
                f"- status: {row.get('status')}",
                f"- hypothesis: {row.get('hypothesis')}",
                f"- mutation: {row.get('mutation_summary')}",
                f"- GAUC: {metrics.get('GAUC', '-')}",
                f"- nDCG@5: {metrics.get('nDCG@5', '-')}",
                f"- primary: {metrics.get('primary', '-')}",
                f"- delta vs parent: {row.get('delta_vs_parent')}",
                f"- delta vs FM: {row.get('delta_vs_fm')}",
                f"- model/thinking: {row.get('model')} / {row.get('thinking_level')}",
                f"- tokens: {row.get('token_counts')}",
                f"- LLM latency (s): {row.get('llm_latency_seconds')}",
                f"- experiment runtime (s): {row.get('experiment_runtime_seconds')}",
                f"- repair calls: {row.get('repair_calls')}",
                f"- error: {row.get('error')}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"
