# Failure recovery matrix

Only behaviors the code actually implements. Tests are FakeProvider / scripted HTTP. No invented retries.

Tokens: counted only when the provider returns a `UsageRecord` into the research ledger. Auth/config failures that raise before a response add **no** experiment to the elite set.

| Failure | Detect | Retry / repair / fail | Tokens counted | Scientific evidence | Can become elite | Test |
| --- | --- | --- | --- | --- | --- | --- |
| Missing `GEMINI_API_KEY` | `resolve_gemini_api_key` | Fail fast. CLI exit 2. No runner. | No | No | No | `test_failure_recovery.py`, `test_llm_provider.py`, `test_evolution_cli.py` |
| HTTP 401 | `GeminiProvider` | `LLMAuthError`. No retry. | No response usage | No | No | `test_llm_provider.py` |
| HTTP 403 | same as 401 | `LLMAuthError`. No retry. | No response usage | No | No | `test_failure_recovery.py` |
| HTTP 429 | status 429 | Up to 3 transport retries, then `LLMRateLimitError` | No success usage | No | No | `test_llm_provider.py` |
| HTTP 5xx / high demand | status ≥ 500 | Up to 3 retries, then `LLMTransientError` | No success usage | No | No | `test_llm_provider.py` |
| HTTP timeout | `TimeoutError` | Retried as transient, then `LLMTransientError` | No success usage | No | No | `test_llm_provider.py` |
| Malformed structured output | parse / protocol | `LLMProtocolError`. Not a silent empty proposal. | Usage may exist on HTTP 200 | No experiment | No | `test_llm_provider.py` |
| Syntax error | `validate_candidate_source` | Repair loop (bounded). Else `implementation_failure`. | Research/repair calls yes | No | No | `test_workspace.py`, `test_research_agent.py` |
| Unsupported dependency | safety scan (`torch`, …) | Reject before subprocess. Repair may rewrite. | Call that proposed it yes | No | No | `test_research_agent.py` |
| Unavailable data field | data-contract check | Claimed aux columns without reading raw CSV → not a fired mechanism / no-op evidence | Yes if a call happened | No (no-op / not tested as claimed) | No | `test_data_contract.py` |
| Invalid / NaN / Inf scores | runner load | `invalid`. No official metrics. | Experiment wall only | No | No | `test_experiment_runner.py`, `test_submission_pack.py` |
| Runtime exception | subprocess non-zero | `failed` / `implementation_failure`. Bounded repair. | Yes for the LLM turn | No | No | `test_experiment_runner.py` |
| Duplicate / no-op copy | diversity + same source/params/seed | Skip execute. Diversity event. | Proposal tokens yes | No | No | `test_evolution_diversity.py`, `test_evolution_controller.py` |
| Incompatible crossover | `crossover_compatible is False` | Fall back to mutation. Not a fake hybrid. | Crossover call yes | Mutation may | Only if mutation succeeds | `test_evolution_controller.py` |
| Budget exhaustion | eval / token / wall / generation / converge | Clean `stop_reason`. No extra offspring. | Up to last counted call | Prior members keep their flags | Prior elites only | `test_competition_budget.py`, `test_evolution_controller.py` |
| Experiment timeout | runner wall | `timeout`. Not elite. | LLM turn yes | No | No | `test_experiment_runner.py` |
| Test split during search | split guard | `ForbiddenTestSplit` / invalid result | n/a | No | No | `test_split_guard.py`, `test_final_candidate.py` |

Live Phase 4: one implementation failure (`np.random.defaultrng`) stayed non-elite. Four scientific negatives stayed in the tree. Zero manual edits.
