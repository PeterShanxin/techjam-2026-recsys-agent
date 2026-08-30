"""Deterministic Evolution Controller. Never invents ML hypotheses."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from research_agent.agent.loop import IterationOutcome, ResearchAgent
from research_agent.agent.root import UnusableRootError, is_usable_root_result
from research_agent.agent.session import experiment_id_for
from research_agent.agent.state import build_research_state
from research_agent.experiments import ExperimentSpec
from research_agent.experiments.canonical import sha256_text
from research_agent.experiments.splits import RESEARCH_SPLIT
from research_agent.llm.secrets import sanitize
from research_agent.llm.types import (
    LLMAuthError,
    LLMConfigError,
    LLMProtocolError,
    LLMRateLimitError,
    LLMTransientError,
)

from .config import EvolutionConfig
from .diversity import duplicate_reason, member_signature
from .fitness import compute_fitness, rank_members, select_elites
from .lineage import format_lineage, lineage_forest
from .prompts import crossover_prompt, mutation_prompt
from .seeds import ensemble_seed_spec
from .types import (
    EvolutionRun,
    GenerationRecord,
    Population,
    PopulationMember,
    SelectionDecision,
)

STOP_GENERATION = "generation_limit"
STOP_EVAL = "evaluation_budget"
STOP_TOKEN = "token_budget"
STOP_WALL = "wall_clock_budget"
STOP_CONVERGED = "converged"
STOP_FATAL = "fatal_provider_error"


class EvolutionController:
    def __init__(self, *, agent: ResearchAgent, config: EvolutionConfig | None = None) -> None:
        self.agent = agent
        self.config = config or EvolutionConfig()
        self.session_id = agent.session_id
        self.trace_dir = Path(agent.runner.runs_dir) / "evolution" / self.session_id
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._slot = 0
        self._evaluated = 0
        self._started = 0.0
        self._last_spawn_kind = "ok"
        self.all_members: list[PopulationMember] = []
        self.diversity_events: list[dict[str, Any]] = []
        self.operator_decisions: list[dict[str, Any]] = []
        self.negative_scientific_hypotheses: list[str] = []
        self.generation_records: list[GenerationRecord] = []
        self.stop_reason: str | None = None
        self.stagnation = 0
        self._best_fitness: float | None = None

    def run(self) -> EvolutionRun:
        self._started = time.perf_counter()
        try:
            population = self._initialize()
            if self.stop_reason is None and self.config.generations == 0:
                self.stop_reason = STOP_GENERATION
            for generation in range(1, self.config.generations + 1):
                if self.stop_reason is not None:
                    break
                reason = self._budget_reason()
                if reason is not None:
                    self.stop_reason = reason
                    break
                population = self._run_generation(generation, population)
                if self.stop_reason is None and self.stagnation >= self.config.convergence_patience:
                    self.stop_reason = STOP_CONVERGED
            if self.stop_reason is None:
                self.stop_reason = STOP_GENERATION
        except (LLMConfigError, LLMAuthError, LLMRateLimitError, LLMTransientError, LLMProtocolError):
            self.stop_reason = STOP_FATAL
            population = Population(members=list(self._current_population_members()))
            self._persist(population)
            raise
        self.agent.ledger.research_wall_seconds = time.perf_counter() - self._started
        run = self._finish(population)
        self._persist(population, run.summary)
        return run

    @classmethod
    def reconstruct(cls, trace_dir: Path, registry: Any) -> EvolutionRun:
        summary = json.loads(Path(trace_dir).joinpath("summary.json").read_text(encoding="utf-8"))
        members = [PopulationMember.from_dict(item) for item in summary.get("population", {}).get("members", [])]
        all_members = [PopulationMember.from_dict(item) for item in summary.get("all_members", [])]
        elites = [PopulationMember.from_dict(item) for item in summary.get("elites", [])]
        return EvolutionRun(
            population=Population(members=members),
            all_members=all_members or members,
            elites=elites,
            generations=[],
            diversity_events=list(summary.get("diversity_events") or []),
            operator_decisions=list(summary.get("operator_decisions") or []),
            negative_scientific_hypotheses=list(summary.get("negative_scientific_hypotheses") or []),
            stop_reason=str(summary.get("stop_reason") or STOP_GENERATION),
            evaluated_offspring=int(summary.get("evaluated_offspring") or 0),
            stagnation=int(summary.get("stagnation") or 0),
            trace_dir=Path(trace_dir),
            summary=summary,
            session_id=str(summary.get("session_id") or ""),
        )

    def _initialize(self) -> Population:
        root = self.agent.ensure_root()
        if not is_usable_root_result(root.result):
            raise UnusableRootError(
                f"research root {root.experiment_id} is not a successful validation result"
            )
        root_member = self._member_from_root(root)
        self.all_members.append(root_member)
        members = [root_member]
        if self.config.include_ensemble_seed:
            seed = self._ensure_ensemble_seed(root_member)
            if seed is not None:
                if seed.experiment_id not in {item.experiment_id for item in self.all_members}:
                    self.all_members.append(seed)
                members.append(seed)
        if self.config.fill_to_size_on_init:
            members = self._fill(members, generation=0)
        population = Population(members=self._mark_elites(members))
        self._record_generation(0, population, decisions=[])
        return population

    def _fill(self, members: list[PopulationMember], *, generation: int) -> list[PopulationMember]:
        attempts = 0
        while len(members) < self.config.population_size and attempts < self.config.population_size * 3:
            attempts += 1
            reason = self._budget_reason()
            if reason is not None:
                self.stop_reason = reason
                break
            spawned = self._spawn("mutation", generation, members)
            if spawned is None:
                if self._last_spawn_kind == "duplicate":
                    continue
                break
            members.append(spawned)
        return members

    def _run_generation(self, generation: int, population: Population) -> Population:
        elites = select_elites(
            population.members,
            self.config.elite_count,
            efficiency_penalty=self.config.efficiency_penalty,
        )
        needed = self.config.offspring_per_generation
        offspring: list[PopulationMember] = []
        decisions: list[dict[str, Any]] = []
        attempts = 0
        while len(offspring) < needed and attempts < needed * 3:
            attempts += 1
            reason = self._budget_reason()
            if reason is not None:
                self.stop_reason = reason
                break
            operator = "mutation"
            if generation >= self.config.prefer_crossover_from_generation:
                pair = _crossover_parents(population.members, self.config.elite_count)
                if pair is not None:
                    operator = "crossover"
            spawned = self._spawn(operator, generation, population.members + offspring)
            if spawned is None:
                if operator == "crossover" or self._last_spawn_kind == "duplicate":
                    continue
                break
            offspring.append(spawned)
            decisions.append({"operator": spawned.origin, "experiment_id": spawned.experiment_id})
        combined = list(elites) + offspring
        seen: set[str] = set()
        unique: list[PopulationMember] = []
        for member in combined:
            if member.experiment_id in seen:
                continue
            seen.add(member.experiment_id)
            unique.append(member)
        next_pop = Population(members=self._mark_elites(unique[: self.config.population_size]))
        self._update_convergence(next_pop)
        self._record_generation(generation, next_pop, decisions)
        return next_pop

    def _spawn(
        self,
        operator: str,
        generation: int,
        population: list[PopulationMember],
    ) -> PopulationMember | None:
        experiment_id = self._next_id()
        if operator == "crossover":
            pair = _crossover_parents(population, self.config.elite_count)
            if pair is None:
                self.operator_decisions.append(
                    SelectionDecision(
                        generation=generation,
                        operator="crossover",
                        parent_ids=(),
                        reason="no_distinct_parents",
                        fallback="mutation",
                    ).to_dict()
                )
                operator = "mutation"
            else:
                left, right = pair
                proposal = self._ask("crossover", generation, population, (left, right), experiment_id)
                if proposal is None:
                    self.operator_decisions.append(
                        SelectionDecision(
                            generation=generation,
                            operator="crossover",
                            parent_ids=(left.experiment_id, right.experiment_id),
                            reason="crossover_proposal_failed",
                            fallback="mutation",
                        ).to_dict()
                    )
                    operator = "mutation"
                    experiment_id = self._next_id()
                elif proposal.crossover_compatible is False:
                    self.operator_decisions.append(
                        SelectionDecision(
                            generation=generation,
                            operator="crossover",
                            parent_ids=(left.experiment_id, right.experiment_id),
                            reason="incompatible_crossover",
                            fallback="mutation",
                        ).to_dict()
                    )
                    operator = "mutation"
                    experiment_id = self._next_id()
                else:
                    return self._execute(
                        proposal,
                        experiment_id=experiment_id,
                        parent_ids=(left.experiment_id, right.experiment_id),
                        origin="crossover",
                        generation=generation,
                    )
        parent = _mutation_parent(population, self._slot, self.config.elite_count)
        self._slot += 1
        proposal = self._ask("mutation", generation, population, (parent,), experiment_id)
        if proposal is None:
            self._last_spawn_kind = "empty"
            return None
        return self._execute(
            proposal,
            experiment_id=experiment_id,
            parent_ids=(parent.experiment_id,),
            origin="mutation",
            generation=generation,
        )

    def _ask(
        self,
        operator: str,
        generation: int,
        population: list[PopulationMember],
        parents: tuple[PopulationMember, ...],
        experiment_id: str,
    ) -> Any:
        dest = self.agent.workspace.dest_for(experiment_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        parent_source = self._source_for(parents[0].experiment_id)
        extras = []
        if operator == "crossover" and len(parents) >= 2:
            extras = [
                {"experiment_id": parents[0].experiment_id, "role": "parent_a", "source": parent_source},
                {
                    "experiment_id": parents[1].experiment_id,
                    "role": "parent_b",
                    "source": self._source_for(parents[1].experiment_id),
                },
            ]
        state = build_research_state(
            registry=self.agent.runner.registry,
            ledger=self.agent.ledger,
            iteration=generation,
            max_iterations=max(self.config.generations, 1),
            remaining_wall_seconds=self._remaining_wall(),
            parent_source=parent_source,
            selected_parent_id=parents[0].experiment_id,
            extra_source_snippets=extras,
            repo_root=self.agent.runner.repo_root,
            data_dir=self.agent.runner.data_dir,
            operator=operator,
            population=[item.to_dict() for item in population],
            crossover_parents=[item.to_dict() for item in parents] if operator == "crossover" else [],
            remaining_evaluation_budget=max(0, self.config.max_new_evaluations - self._evaluated),
            remaining_token_budget=(
                None
                if self.config.token_budget is None
                else max(0, self.config.token_budget - self.agent.ledger.total_tokens)
            ),
            remaining_generations=max(0, self.config.generations - generation),
        )
        prompt = crossover_prompt(state) if operator == "crossover" else mutation_prompt(state)
        proposal, _usages, _error = self.agent.propose_candidate(
            state,
            dest,
            purpose=operator,
            prompt=prompt,
        )
        return proposal

    def _execute(
        self,
        proposal: Any,
        *,
        experiment_id: str,
        parent_ids: tuple[str, ...],
        origin: str,
        generation: int,
    ) -> PopulationMember | None:
        source_fp = sha256_text(proposal.candidate_source)
        parent_sources = [self._source_for(pid) for pid in parent_ids]
        probe = PopulationMember(
            experiment_id="pending",
            parent_ids=parent_ids,
            generation=generation,
            origin=origin,
            hypothesis=proposal.hypothesis,
            rationale=proposal.rationale,
            research_family=proposal.research_family or "other",
            mechanism_tags=tuple(proposal.mechanism_tags),
            changed_axes=tuple(proposal.changed_axes),
            source_fingerprint=source_fp,
            spec_hash="",
            metrics=None,
            research_validity="not_executed",
            runtime_seconds=None,
            resource_usage={},
            status="invalid",
            evaluation_split=RESEARCH_SPLIT,
            selection="rejected_duplicate",
            scientific_evidence=False,
        )
        reason = duplicate_reason(probe, self.all_members)
        matching_source = [
            item
            for item in self.all_members
            if probe.source_fingerprint and item.source_fingerprint == probe.source_fingerprint
        ]
        only_baseline_parent_copy = bool(matching_source) and all(
            item.origin == "baseline" or item.research_validity == "root" for item in matching_source
        )
        same_params_as_source_twin = self._same_params_and_seed(proposal, matching_source)
        suppress_duplicate = False
        if reason == "semantic_signature" or reason == "spec_hash":
            suppress_duplicate = True
        elif reason == "source_fingerprint" and same_params_as_source_twin and not only_baseline_parent_copy:
            suppress_duplicate = True
        if suppress_duplicate:
            self._last_spawn_kind = "duplicate"
            self.diversity_events.append(
                {
                    "reason": reason,
                    "hypothesis": proposal.hypothesis,
                    "research_family": probe.research_family,
                    "generation": generation,
                }
            )
            return None
        if reason == "source_fingerprint":
            self.diversity_events.append(
                {
                    "reason": "source_fingerprint",
                    "hypothesis": proposal.hypothesis,
                    "research_family": probe.research_family,
                    "generation": generation,
                    "detail": "parent_copy" if same_params_as_source_twin else "parameter_only",
                }
            )
        self._last_spawn_kind = "ok"
        dest = self.agent.workspace.dest_for(experiment_id)
        parent_source = parent_sources[0]
        materialized = self.agent.workspace.materialize(
            experiment_id=experiment_id,
            source=proposal.candidate_source,
            parent_source=parent_source,
            repo_root=self.agent.runner.repo_root,
        )
        (dest.parent / "proposal.json").write_text(proposal.to_json(), encoding="utf-8")
        spec = ExperimentSpec(
            experiment_id=experiment_id,
            implementation=materialized.implementation,
            hypothesis=proposal.hypothesis,
            rationale=proposal.rationale,
            origin=origin,
            parent_ids=parent_ids,
            parameters=dict(proposal.experiment_parameters),
            seed=proposal.seed,
            evaluation_split=RESEARCH_SPLIT,
            timeout_seconds=min(proposal.timeout_seconds, self.config.experiment_timeout_seconds),
            allow_test_split=False,
            tags=_research_tags(self.session_id, generation, proposal),
            notes=_research_notes(generation, proposal),
        )
        result = self.agent.runner.run(spec)
        self.agent.ledger.add_experiment(status=result.status, wall_seconds=result.wall_seconds)
        self._evaluated += 1
        parent_source_all = parent_sources
        parent_entry = self.agent.runner.registry.peek(parent_ids[0]) if parent_ids else None
        parent_spec = None if parent_entry is None else parent_entry.spec
        validity = _classify_validity(
            result.status,
            proposal.candidate_source,
            parent_source_all,
            executed=True,
            child_parameters=dict(proposal.experiment_parameters),
            child_seed=proposal.seed,
            parent_parameters=dict(parent_spec.parameters) if parent_spec is not None else {},
            parent_seed=None if parent_spec is None else parent_spec.seed,
        )
        if validity == "semantic_noop":
            self.agent.runner.registry.mark_decision(experiment_id, "rejected")
        scientific = validity == "hypothesis_tested" and result.status == "success"
        member = self._member_from_result(
            spec,
            result,
            generation=generation,
            origin=origin,
            proposal=proposal,
            source_fingerprint=source_fp,
            research_validity=validity,
            scientific_evidence=scientific,
        )
        member = member.with_updates(
            fitness=compute_fitness(member, efficiency_penalty=self.config.efficiency_penalty)
        )
        self.all_members.append(member)
        if scientific:
            parent_fit = _parent_primary(self.all_members, parent_ids[0])
            if (
                parent_fit is not None
                and member.metrics is not None
                and float(member.metrics["primary"]) < parent_fit
            ):
                self.negative_scientific_hypotheses.append(proposal.hypothesis)
        self.operator_decisions.append(
            SelectionDecision(
                generation=generation,
                operator=origin,
                parent_ids=parent_ids,
                reason="evaluated",
                experiment_id=experiment_id,
            ).to_dict()
        )
        return member

    def _ensure_ensemble_seed(self, root: PopulationMember) -> PopulationMember | None:
        spec = ensemble_seed_spec(parent_id=root.experiment_id)
        existing = self.agent.runner.registry.peek(spec.experiment_id)
        if existing is not None and existing.result is not None:
            return self._member_from_result(
                existing.spec,
                existing.result,
                generation=0,
                origin="mutation",
                proposal=None,
                source_fingerprint=sha256_text(self._source_for(existing.spec.experiment_id)),
                research_validity="hypothesis_tested" if existing.result.status == "success" else "implementation_failure",
                scientific_evidence=existing.result.status == "success",
            )
        result = self.agent.runner.run(spec)
        self.agent.ledger.add_experiment(status=result.status, wall_seconds=result.wall_seconds)
        member = self._member_from_result(
            spec,
            result,
            generation=0,
            origin="mutation",
            proposal=None,
            source_fingerprint=sha256_text(self._source_for(spec.experiment_id)),
            research_validity="hypothesis_tested" if result.status == "success" else "implementation_failure",
            scientific_evidence=result.status == "success",
        )
        member = member.with_updates(
            fitness=compute_fitness(member, efficiency_penalty=self.config.efficiency_penalty),
            research_family="ensemble",
            mechanism_tags=("bagging",),
            changed_axes=("ensembling",),
        )
        return member

    def _member_from_root(self, outcome: IterationOutcome) -> PopulationMember:
        spec = self.agent.runner.registry.get(outcome.experiment_id).spec
        result = outcome.result
        source = self._source_for(outcome.experiment_id)
        member = self._member_from_result(
            spec,
            result,
            generation=0,
            origin="baseline",
            proposal=None,
            source_fingerprint=sha256_text(source),
            research_validity="root",
            scientific_evidence=True,
        )
        return member.with_updates(
            fitness=compute_fitness(member, efficiency_penalty=self.config.efficiency_penalty),
            research_family="factorization_machine",
            parent_ids=(),
        )

    def _member_from_result(
        self,
        spec: ExperimentSpec,
        result: Any,
        *,
        generation: int,
        origin: str,
        proposal: Any,
        source_fingerprint: str,
        research_validity: str,
        scientific_evidence: bool,
    ) -> PopulationMember:
        metrics = None
        if result is not None and getattr(result, "metrics", None) is not None:
            metrics = {
                "GAUC": float(result.metrics.gauc),
                "nDCG@5": float(result.metrics.ndcg_at_5),
                "primary": float(result.metrics.primary),
            }
        family = (proposal.research_family if proposal is not None else "") or _family_from_tags(spec.tags)
        if origin == "baseline" and not family:
            family = "factorization_machine"
        tags = tuple(proposal.mechanism_tags) if proposal is not None else _tag_values(spec.tags, "mech:")
        axes = tuple(proposal.changed_axes) if proposal is not None else _tag_values(spec.tags, "axis:")
        status = "invalid" if result is None else result.status
        return PopulationMember(
            experiment_id=spec.experiment_id,
            parent_ids=tuple(spec.parent_ids),
            generation=generation,
            origin=origin,
            hypothesis=spec.hypothesis if proposal is None else proposal.hypothesis,
            rationale=spec.rationale if proposal is None else proposal.rationale,
            research_family=family or "other",
            mechanism_tags=tags,
            changed_axes=axes,
            source_fingerprint=source_fingerprint,
            spec_hash=spec.spec_hash,
            metrics=metrics,
            research_validity=research_validity,
            runtime_seconds=None if result is None else result.wall_seconds,
            resource_usage={"wall_seconds": None if result is None else result.wall_seconds},
            status=status,
            evaluation_split=spec.evaluation_split,
            selection="pending",
            scientific_evidence=scientific_evidence,
        )

    def _mark_elites(self, members: list[PopulationMember]) -> list[PopulationMember]:
        elites = select_elites(
            members, self.config.elite_count, efficiency_penalty=self.config.efficiency_penalty
        )
        elite_ids = {item.experiment_id for item in elites}
        marked_elites = []
        marked_rest = []
        by_id = {item.experiment_id: item for item in members}
        for elite in elites:
            member = by_id[elite.experiment_id]
            marked_elites.append(member.with_updates(selection="elite", fitness=elite.fitness))
        for member in members:
            if member.experiment_id in elite_ids:
                continue
            selection = "active"
            if member.research_validity == "semantic_noop":
                selection = "rejected_noop"
            elif not member.scientific_evidence and member.origin != "baseline":
                if member.status != "success":
                    selection = "rejected_invalid"
            marked_rest.append(member.with_updates(selection=selection))
        return marked_elites + marked_rest

    def _update_convergence(self, population: Population) -> None:
        ranked = rank_members(population.members, efficiency_penalty=self.config.efficiency_penalty)
        current = None if not ranked else ranked[0].fitness
        if current is None:
            self.stagnation += 1
            return
        if self._best_fitness is None:
            self._best_fitness = current
            self.stagnation = 0
            return
        improvement = current - self._best_fitness
        if improvement > self.config.convergence_epsilon:
            self._best_fitness = current
            self.stagnation = 0
        else:
            self.stagnation += 1

    def _record_generation(
        self, generation: int, population: Population, decisions: list[dict[str, Any]]
    ) -> None:
        elites = [
            item.experiment_id
            for item in select_elites(
                population.members,
                self.config.elite_count,
                efficiency_penalty=self.config.efficiency_penalty,
            )
        ]
        ranked = rank_members(population.members, efficiency_penalty=self.config.efficiency_penalty)
        best = None if not ranked else ranked[0].fitness
        prev = None if not self.generation_records else self.generation_records[-1].best_fitness
        improvement = None if best is None or prev is None else best - prev
        self.generation_records.append(
            GenerationRecord(
                generation=generation,
                member_ids=[item.experiment_id for item in population.members],
                elite_ids=elites,
                best_fitness=best,
                improvement=improvement,
                stagnation=self.stagnation,
                decisions=decisions,
                stop_reason=self.stop_reason,
            )
        )

    def _budget_reason(self) -> str | None:
        if self._evaluated >= self.config.max_new_evaluations:
            return STOP_EVAL
        if (
            self.config.token_budget is not None
            and self.agent.ledger.total_tokens >= self.config.token_budget
        ):
            return STOP_TOKEN
        remaining = self._remaining_wall()
        if remaining is not None and remaining <= 0:
            return STOP_WALL
        return None

    def _remaining_wall(self) -> float | None:
        if self.config.wall_clock_seconds is None:
            return self.agent._remaining_wall(self._started) if self._started else self.agent.wall_clock_seconds
        return max(0.0, float(self.config.wall_clock_seconds) - (time.perf_counter() - self._started))

    def _next_id(self) -> str:
        self._seq += 1
        return experiment_id_for(self.session_id, self._seq)

    def _source_for(self, experiment_id: str) -> str:
        spec = self.agent.runner.registry.get(experiment_id).spec
        return self.agent.workspace.load_parent_source(spec, self.agent.runner.repo_root)

    def _same_params_and_seed(self, proposal: Any, members: list[PopulationMember]) -> bool:
        child_params = dict(getattr(proposal, "experiment_parameters", {}) or {})
        child_seed = getattr(proposal, "seed", None)
        for item in members:
            entry = self.agent.runner.registry.peek(item.experiment_id)
            if entry is None:
                continue
            if dict(entry.spec.parameters) == child_params and entry.spec.seed == child_seed:
                return True
        return False

    def _current_population_members(self) -> list[PopulationMember]:
        if not self.generation_records:
            return list(self.all_members)
        ids = set(self.generation_records[-1].member_ids)
        latest = {item.experiment_id: item for item in self.all_members}
        return [latest[item] for item in self.generation_records[-1].member_ids if item in latest] or list(
            self.all_members
        )

    def _finish(self, population: Population) -> EvolutionRun:
        elites = select_elites(
            population.members,
            self.config.elite_count,
            efficiency_penalty=self.config.efficiency_penalty,
        )
        forest = lineage_forest(self.agent.runner.registry)
        tree = format_lineage(forest)
        (self.trace_dir / "tree.txt").write_text(tree, encoding="utf-8")
        summary = sanitize(
            {
                "session_id": self.session_id,
                "stop_reason": self.stop_reason,
                "evaluated_offspring": self._evaluated,
                "stagnation": self.stagnation,
                "population": population.to_dict(),
                "all_members": [item.to_dict() for item in self.all_members],
                "elites": [item.to_dict() for item in elites],
                "diversity_events": list(self.diversity_events),
                "operator_decisions": list(self.operator_decisions),
                "negative_scientific_hypotheses": list(self.negative_scientific_hypotheses),
                "generations": [item.to_dict() for item in self.generation_records],
                "lineage": tree,
                "resources": self.agent.ledger.to_dict(),
                "manual_interventions": self.agent.ledger.manual_interventions,
                "config": {
                    "population_size": self.config.population_size,
                    "elite_count": self.config.elite_count,
                    "generations": self.config.generations,
                    "max_new_evaluations": self.config.max_new_evaluations,
                },
            }
        )
        return EvolutionRun(
            population=population,
            all_members=list(self.all_members),
            elites=elites,
            generations=list(self.generation_records),
            diversity_events=list(self.diversity_events),
            operator_decisions=list(self.operator_decisions),
            negative_scientific_hypotheses=list(self.negative_scientific_hypotheses),
            stop_reason=self.stop_reason or STOP_GENERATION,
            evaluated_offspring=self._evaluated,
            stagnation=self.stagnation,
            trace_dir=self.trace_dir,
            summary=summary,
            session_id=self.session_id,
        )

    def _persist(self, population: Population, summary: dict[str, Any] | None = None) -> None:
        payload = summary or self._finish(population).summary
        (self.trace_dir / "summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        (self.trace_dir / "population.json").write_text(
            json.dumps(population.to_dict(), indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        with (self.trace_dir / "generations.jsonl").open("w", encoding="utf-8") as handle:
            for record in self.generation_records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=True) + "\n")


def _crossover_parents(
    members: list[PopulationMember], elite_count: int
) -> tuple[PopulationMember, PopulationMember] | None:
    ranked = rank_members(members)
    if len(ranked) < 2:
        return None
    left = ranked[0]
    for right in ranked[1:]:
        if member_signature(left) != member_signature(right):
            return left, right
    return None


def _mutation_parent(
    members: list[PopulationMember], slot: int, elite_count: int
) -> PopulationMember:
    elites = select_elites(members, elite_count) or list(members)
    return elites[slot % len(elites)]


def _classify_validity(
    status: str,
    source: str,
    parent_sources: list[str],
    *,
    executed: bool,
    child_parameters: dict[str, Any] | None = None,
    child_seed: int | None = None,
    parent_parameters: dict[str, Any] | None = None,
    parent_seed: int | None = None,
) -> str:
    if not executed:
        return "not_executed"
    if status != "success":
        return "implementation_failure"
    stripped = source.strip()
    same_source = any(stripped == parent.strip() for parent in parent_sources if parent)
    if not same_source:
        return "hypothesis_tested"
    same_params = (child_parameters or {}) == (parent_parameters or {})
    same_seed = child_seed == parent_seed
    if same_params and same_seed:
        return "semantic_noop"
    return "hypothesis_tested"


def _research_tags(session_id: str, generation: int, proposal: Any) -> tuple[str, ...]:
    tags = ["phase4", "autonomous", session_id, f"gen:{generation}"]
    if proposal.research_family:
        tags.append(f"family:{proposal.research_family}")
    for axis in proposal.changed_axes:
        tags.append(f"axis:{axis}")
    for mech in proposal.mechanism_tags:
        tags.append(f"mech:{mech}")
    return tuple(tags)


def _research_notes(generation: int, proposal: Any) -> str:
    return json.dumps(
        {
            "generation": generation,
            "operator": proposal.operator,
            "research_family": proposal.research_family,
            "mechanism_tags": list(proposal.mechanism_tags),
            "changed_axes": list(proposal.changed_axes),
            "what_changed": proposal.what_changed,
            "why": proposal.why,
            "evidence_motivated": proposal.evidence_motivated,
            "would_support": proposal.would_support,
            "would_refute": proposal.would_refute,
            "parent_a_component": proposal.parent_a_component,
            "parent_b_component": proposal.parent_b_component,
            "crossover_compatible": proposal.crossover_compatible,
        },
        sort_keys=True,
        ensure_ascii=True,
    )


def _family_from_tags(tags: tuple[str, ...]) -> str:
    for tag in tags:
        if tag.startswith("family:"):
            return tag.split(":", 1)[1]
    return ""


def _tag_values(tags: tuple[str, ...], prefix: str) -> tuple[str, ...]:
    return tuple(tag.split(":", 1)[1] for tag in tags if tag.startswith(prefix))


def _parent_primary(members: list[PopulationMember], parent_id: str) -> float | None:
    for member in members:
        if member.experiment_id == parent_id and member.metrics and "primary" in member.metrics:
            return float(member.metrics["primary"])
    return None
