"""Run the sequential Research Agent.

python scripts/run_research_agent.py --iterations 3 --model gemini-3.7-flash --thinking medium
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from research_agent.agent import ResearchAgent, UnusableRootError
from research_agent.agent.constants import DEFAULT_RESEARCH_MODEL, DEFAULT_THINKING_LEVEL
from research_agent.evolution.seeds import ensure_matched_starting_seeds
from research_agent.experiments import ExperimentRunner
from research_agent.llm import (
    FakeProvider,
    GeminiProvider,
    LLMAuthError,
    LLMConfigError,
    LLMProtocolError,
    LLMRateLimitError,
    LLMTransientError,
)
from research_agent.llm.credentials import resolve_gemini_api_key
from research_agent.llm.secrets import redact_text
from research_agent.llm.types import THINKING_LEVELS, normalize_thinking_level


def _resolve_data_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("KUAI_RAND_DATA_DIR")
    if env:
        return Path(env)
    return ROOT / "starter" / "kuairand" / "KuaiRand-Pure" / "data"


def _print_event(event: dict) -> None:
    kind = event.get("type")
    if kind == "root":
        reused = " (reused)" if event.get("reused") else ""
        print(f"FM root {event.get('experiment_id')} status={event.get('status')}{reused}")
        return
    if kind == "iteration":
        print("")
        print(
            f"=== session {event.get('session_id')} "
            f"iteration {event.get('iteration')}/{event.get('max_iterations')} "
            f"parent={event.get('parent')} elite={event.get('elite')} ==="
        )
        return
    if kind == "repair":
        print(
            f"repair {event.get('attempt')} thinking={event.get('thinking_level')} "
            f"error={redact_text(str(event.get('error') or ''))}"
        )
        return
    if kind == "result":
        metrics_line = (
            f"GAUC={_fmt(event.get('GAUC'))} nDCG@5={_fmt(event.get('ndcg_at_5'))} "
            f"primary={_fmt(event.get('primary'))} delta={_fmt(event.get('delta_vs_parent'))}"
        )
        tokens = event.get("tokens") or {}
        print(f"hypothesis  {event.get('hypothesis') or '-'}")
        print(f"status      {event.get('status')}")
        print(metrics_line)
        print(
            f"tokens      in={tokens.get('input_tokens')} out={tokens.get('output_tokens')} "
            f"think={tokens.get('thinking_tokens')} total={tokens.get('total_tokens')} "
            f"thinking={event.get('thinking_level')}"
        )
        print(
            f"repairs     {event.get('repair_calls')} remaining_experiments={event.get('remaining_experiments')}"
        )
        if event.get("error"):
            print(f"error       {redact_text(str(event.get('error')))}")
        return
    if kind == "done":
        summary = event.get("summary") or {}
        resources = summary.get("resources") or {}
        print("")
        print("=== done ===")
        print(f"session     {summary.get('session_id')}")
        print(f"best        {summary.get('best')}")
        print(f"improvement vs FM {summary.get('improvement_vs_fm')}")
        print(
            f"LLM calls   {resources.get('llm_calls')} "
            f"(research={resources.get('research_calls')} repair={resources.get('repair_calls')})"
        )
        print(
            f"tokens      in={resources.get('input_tokens')} out={resources.get('output_tokens')} "
            f"think={resources.get('thinking_tokens')} total={resources.get('total_tokens')}"
        )
        print(f"manual interventions {resources.get('manual_interventions')}")
        return
    if kind == "budget":
        print(f"stop: {event.get('reason')} at iteration {event.get('iteration')}")


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sequential autonomous KuaiRand research loop.")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--model", default=DEFAULT_RESEARCH_MODEL)
    ap.add_argument("--thinking", default=DEFAULT_THINKING_LEVEL, choices=list(THINKING_LEVELS))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--wall-clock", type=float, default=None)
    ap.add_argument("--max-repairs", type=int, default=2)
    ap.add_argument("--manual-interventions", type=int, default=0)
    ap.add_argument(
        "--with-ensemble-prior",
        action="store_true",
        help="Insert verified fm-ensemble-3seed before sequential search. Does not count as a new evaluation.",
    )
    ap.add_argument(
        "--provider",
        default="gemini",
        choices=("gemini", "fake"),
        help="fake is for dry tests only",
    )
    args = ap.parse_args(argv)

    thinking = normalize_thinking_level(args.thinking)
    if args.provider == "fake":
        provider = FakeProvider(script=[])
        print("provider    fake (no API calls)")
    else:
        try:
            resolve_gemini_api_key(ROOT)
        except LLMConfigError as exc:
            print(redact_text(str(exc)), file=sys.stderr)
            return 2
        provider = GeminiProvider(model=args.model, repo_root=ROOT)

    data_dir = _resolve_data_dir(args.data_dir)
    runs_dir = Path(args.runs_dir) if args.runs_dir else ROOT / "runs"
    runner = ExperimentRunner(
        repo_root=ROOT,
        runs_dir=runs_dir,
        data_dir=data_dir,
        allow_test=False,
    )
    print(f"model       {args.model}")
    print(f"thinking    {thinking}")
    print(f"iterations  {args.iterations}")
    print(f"data        {data_dir}")
    print(f"runs        {runs_dir}")

    agent = ResearchAgent(
        provider=provider,
        runner=runner,
        model=args.model,
        thinking_level=thinking,
        max_iterations=args.iterations,
        max_repairs=args.max_repairs,
        wall_clock_seconds=args.wall_clock,
        manual_interventions=args.manual_interventions,
        emit=_print_event,
        experiment_timeout_seconds=args.timeout,
    )
    print(f"session     {agent.session_id}")
    try:
        if args.with_ensemble_prior:
            print("priors      fm-root + fm-ensemble-3seed (not counted as new evaluations)")
            ensure_matched_starting_seeds(agent)
        run = agent.run()
    except (LLMConfigError, LLMAuthError, LLMRateLimitError, LLMTransientError, LLMProtocolError, UnusableRootError) as exc:
        print(redact_text(str(exc)), file=sys.stderr)
        return 2
    ok = True
    if run.root is None or run.root.result_status != "success":
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
