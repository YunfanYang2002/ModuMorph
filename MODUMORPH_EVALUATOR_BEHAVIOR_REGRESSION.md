# ModuMorph evaluator behavior regression

## Frozen claim and case selection

This regression tests behavior equivalence rather than source-byte equality. The runner reads the existing canonical `canonical_runs.tsv`, requires training seed 1409 and method `state_action`, and validates the recorded checkpoint and config SHA256. It then binds the unique rendered mutation command to its existing raw result by content and path, without guessing from the result filename.

The raw result must record evaluation seed 1409, contain protocol `ood_strong`, enable mutation at step 250, and match the canonical checkpoint/config hashes. The first Strict-OOD97 walker with one episode and a nonempty recovery/adaptation event is selected. If those event fields cannot prove mutation inclusion, authoritative episode length at least 250 is the minimum selection gate and is recorded as the selection basis. Missing episode length fails loudly.

No historical trace is required and no new reference is generated. The selected walker must have exactly one row in the bound 97-walker/87-cluster `strict_ood97_identity.tsv`, and its runtime XML must match the recorded SHA256. The reference record stores the raw result, rendered command, checkpoint, config and walker XML paths and hashes.

## Candidate execution

The runner reconstructs commit `3893388b79b0df4b10ad3cd8504258618947325c` as a detached clean worktree. It verifies all 29 evaluator/config source hashes before execution, then invokes that commit's original `tools/evaluate_dynamics.py` using the reference checkpoint, config and walker asset. The frozen case is one episode, one walker, evaluation seed 1409, `ood_strong`, mutation step 250, 1000-step maximum, and the formally rendered recovery and mutation ranges.

## Acceptance contract

The primary comparison recursively covers the entire authoritative `per_walker[walker]` raw record. Dictionary keys, list lengths, null structure, booleans, integers, episode/event counts, episode lengths and discrete recovery/relapse fields are exact. Every float in that record uses fixed `rtol=1e-10`, `atol=1e-10`, including return, PNP, NAR, RMRT, retention, normalized drop and recovery summaries when present.

The candidate trace must prove that mutation actually fired at step 250 and that the trace reached that transition. If an explicitly supplied historical trace matches the same canonical identity, trace-level reward, velocity, action and other continuous fields are additionally compared at the same fixed tolerance. Its absence produces `TRACE_LEVEL_REGRESSION=NOT_AVAILABLE` and does not block raw regression. A raw failure reports the first recursive path plus actual, expected, absolute difference and relative difference. A PASS reports maximum absolute and relative differences.

## Current evidence status

Local tests validate source-gate behavior and comparison rules using fixtures. No MuJoCo replay was run locally, and Codex did not access the server.

```text
REFERENCE_AUTHORITY=CANONICAL_TABLE2_RAW_RESULT
BEHAVIOR_REGRESSION_REFERENCE=NOT_SELECTED_SERVER_REQUIRED
PREVIOUS_REGRESSION_ATTEMPT=REFERENCE_SELECTION_BLOCKED
BEHAVIOR_REPLAY_EXECUTED=NO
REPRODUCIBLE_CANDIDATE_SHA=3893388b79b0df4b10ad3cd8504258618947325c
FROZEN_EVALUATOR_BEHAVIOR_REGRESSION=NOT_RUN
BINDING_CONTRACT_REPAIRED=NO
SOURCE_INTEGRITY_GATE_PRESERVED=YES
MODUMORPH_245K_SMOKE_READY=NO
FORMAL_MODUMORPH_EVAL_READY=NO
```

From the activated server environment, run only:

```bash
bash scripts/run_modumorph_evaluator_behavior_regression_server.sh 2 --no-keep-open
```

The default reference is the adjacent rmamorph `tmp/morphadapt_canonical_student_formal_table2_20260905T111437Z`. Historical trace comparison is disabled by default and is not required. Success requires `FROZEN_EVALUATOR_BEHAVIOR_REGRESSION=PASS` and `RUN_EXIT_CODE=0`. Return the printed `OUTPUT_ZIP` for review before changing the binding or running the ModuMorph 245K smoke.
