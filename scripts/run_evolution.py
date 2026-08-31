"""Run LLM-guided evolutionary search. Controller is deterministic. Gemini only mutates/crosses."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from research_agent.agent import ResearchAgent, UnusableRootError
from research_agent.agent.constants import DEFAULT_RESEARCH_MODEL, DEFAULT_THINKING_LEVEL
from research_agent.evolution import EvolutionConfig, EvolutionController
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
    if kind == "repair":
        print(
            f"repair {event.get('attempt')} thinking={event.get('thinking_level')} "
            f"error={redact_text(str(event.get('error') or ''))}"
        )
        return
    if kind == "done":
        summary = event.get("summary") or {}
        print(f"sequential done session={summary.get('session_id')} best={summary.get('best')}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evolutionary KuaiRand research search.")
    ap.add_argument("--model", default=DEFAULT_RESEARCH_MODEL)
    ap.add_argument("--thinking", default=DEFAULT_THINKING_LEVEL, choices=list(THINKING_LEVELS))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--wall-clock", type=float, default=None)
    ap.add_argument("--max-repairs", type=int, default=2)
    ap.add_argument("--manual-interventions", type=int, default=0)
    ap.add_argument("--population-size", type=int, default=4)
    ap.add_argument("--elite-count", type=int, default=2)
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--max-new-evaluations", type=int, default=6)
    ap.add_argument("--token-budget", type=int, default=None)
    ap.add_argument("--no-ensemble-seed", action="store_true")
    ap.add_argument("--no-fill", action="store_true")
    ap.add_argument("--sequential-control", action="store_true")
    ap.add_argument(
        "--competition",
        action="store_true",
        help="Official Track 2 budgets: 50 new evaluations, 6h wall, epsilon=0.002, patience=3",
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
    print(f"data        {data_dir}")
    print(f"runs        {runs_dir}")
    if args.competition:
        config = EvolutionConfig.competition(
            population_size=args.population_size,
            elite_count=args.elite_count,
            include_ensemble_seed=not args.no_ensemble_seed,
            fill_to_size_on_init=not args.no_fill,
            token_budget=args.token_budget,
            wall_clock_seconds=args.wall_clock if args.wall_clock is not None else 21600.0,
            experiment_timeout_seconds=args.timeout,
            max_repairs=args.max_repairs,
        )
        agent_wall = config.wall_clock_seconds
        agent_iters = config.max_new_evaluations
    else:
        config = EvolutionConfig(
            population_size=args.population_size,
            elite_count=args.elite_count,
            generations=args.generations,
            max_new_evaluations=args.max_new_evaluations,
            include_ensemble_seed=not args.no_ensemble_seed,
            fill_to_size_on_init=not args.no_fill,
            token_budget=args.token_budget,
            wall_clock_seconds=args.wall_clock,
            experiment_timeout_seconds=args.timeout,
            max_repairs=args.max_repairs,
        )
        agent_wall = args.wall_clock
        agent_iters = args.max_new_evaluations

    print(
        f"population  {config.population_size} elite={config.elite_count} "
        f"generations={config.generations} max_new={config.max_new_evaluations}"
    )
    if args.competition:
        print("budget      competition (50 evals, 6h, epsilon=0.002, patience=3)")

    agent = ResearchAgent(
        provider=provider,
        runner=runner,
        model=args.model,
        thinking_level=thinking,
        max_iterations=agent_iters,
        max_repairs=args.max_repairs,
        wall_clock_seconds=agent_wall,
        manual_interventions=args.manual_interventions,
        emit=_print_event,
        experiment_timeout_seconds=args.timeout,
    )
    print(f"session     {agent.session_id}")
    controller = EvolutionController(agent=agent, config=config)
    try:
        run = controller.run()
    except (LLMConfigError, LLMAuthError, LLMRateLimitError, LLMTransientError, LLMProtocolError, UnusableRootError) as exc:
        print(redact_text(str(exc)), file=sys.stderr)
        return 2

    print("")
    print("=== evolution done ===")
    print(f"stop        {run.stop_reason}")
    print(f"evaluated   {run.evaluated_offspring}")
    print(f"elites      {[item.experiment_id for item in run.elites]}")
    if run.elites:
        best = run.elites[0]
        print(f"best        {best.experiment_id} fitness={best.fitness} primary={(best.metrics or {}).get('primary')}")
    print(f"trace       {run.trace_dir}")
    print(run.summary.get("lineage") or "")

    if args.sequential_control:
        control_n = max(1, run.evaluated_offspring)
        print("")
        print(f"=== sequential control ({control_n} iterations, independent registry) ===")
        print("priors      fm-root + fm-ensemble-3seed (not counted as new evaluations)")
        control_dir = runs_dir / "sequential-control"
        control_runner = ExperimentRunner(
            repo_root=ROOT,
            runs_dir=control_dir,
            data_dir=data_dir,
            allow_test=False,
        )
        seq = ResearchAgent(
            provider=provider,
            runner=control_runner,
            model=args.model,
            thinking_level=thinking,
            max_iterations=control_n,
            max_repairs=args.max_repairs,
            wall_clock_seconds=agent_wall,
            manual_interventions=0,
            emit=_print_event,
            experiment_timeout_seconds=args.timeout,
        )
        try:
            ensure_matched_starting_seeds(seq)
            seq_run = seq.run()
        except (LLMConfigError, LLMAuthError, LLMRateLimitError, LLMTransientError, LLMProtocolError, UnusableRootError) as exc:
            print(redact_text(str(exc)), file=sys.stderr)
            return 2
        compare = {
            "starting_seeds": ["fm-root", "fm-ensemble-3seed"],
            "new_evaluations": control_n,
            "evolution": {
                "best": None if not run.elites else run.elites[0].to_dict(),
                "evaluated": run.evaluated_offspring,
                "tokens": run.summary.get("resources"),
                "families": sorted(
                    {item.research_family for item in run.all_members if item.research_family}
                ),
            },
            "sequential": seq_run.summary,
        }
        out = run.trace_dir / "sequential_control.json"
        out.write_text(json.dumps(compare, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"compare     {out}")

    ok = run.stop_reason != "fatal_provider_error"
    if run.population.members and all(item.status != "success" for item in run.population.members):
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
