# ModuMorph evaluator behavior regression

## Frozen claim and case selection

This regression tests behavior equivalence rather than source-byte equality. The runner reads the existing canonical `canonical_runs.tsv`, requires training seed 1409 and method `state_action`, validates the recorded checkpoint and config SHA256, and searches an authoritative existing trace root. It selects the first deterministic Strict-OOD97 episode satisfying all of:

- evaluation seed 1409;
- protocol `ood_strong` and setting `mutation`;
- mutation recorded at step 250;
- episode length and trace length at least 250 and mutually equal;
- trace checkpoint/config hashes equal the canonical manifest.

An early-terminated trace that did not reach the mutation is ineligible. No new reference is generated. The selected walker must have exactly one row in `strict_ood97_identity.tsv`, and its runtime XML must match the recorded SHA256.

## Candidate execution

The runner reconstructs commit `3893388b79b0df4b10ad3cd8504258618947325c` as a detached clean worktree. It verifies all 29 evaluator/config source hashes before execution, then invokes that commit's original `tools/evaluate_dynamics.py` using the reference checkpoint, config and walker asset. The frozen case is one episode, one walker, evaluation seed 1409, `ood_strong`, mutation step 250, 1000-step maximum, and the formally rendered recovery and mutation ranges.

## Acceptance contract

The following are exact: walker/seed/protocol/setting/checkpoint/config identity, episode length, number of trace steps, all dictionary keys and null structure, mutation record and sampled parameters, timestep, mutation flags, termination/truncation flags, strings, integers and booleans.

Floating trace values use fixed `rtol=1e-10`, `atol=1e-10`: reward, measured forward velocity, recovery performance, policy action, Student latent/action-history fields when present, and any recorded physical context. Formal per-walker result data and episode recovery/adaptation metrics use the same fixed tolerance for floats while preserving all discrete and null structure. A failure reports the first exact recursive path plus actual, expected, absolute difference and relative difference. A PASS reports maximum absolute and relative differences by field group.

## Current evidence status

Local tests validate source-gate behavior and comparison rules using fixtures. No MuJoCo replay was run locally, and Codex did not access the server.

```text
BEHAVIOR_REGRESSION_REFERENCE=NOT_SELECTED_SERVER_REQUIRED
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

The default reference is the adjacent rmamorph `tmp/morphadapt_canonical_student_formal_table2_20260905T111437Z`. If its authoritative trace evidence is stored separately, pass `--trace-root <AUTHORITATIVE_EXISTING_TRACE_ROOT>`; do not point this option at a newly generated or diagnostic trace. Success requires `FROZEN_EVALUATOR_BEHAVIOR_REGRESSION=PASS` and `RUN_EXIT_CODE=0`. Return the printed `OUTPUT_ZIP` for review before changing the binding or running the ModuMorph 245K smoke.
