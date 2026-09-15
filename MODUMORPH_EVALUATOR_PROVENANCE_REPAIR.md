# ModuMorph frozen evaluator provenance repair

## Established provenance defect

`configs/modumorph_frozen_eval_binding.json` records commit `3893388b79b0df4b10ad3cd8504258618947325c` and 29 normalized source hashes. The commit reconstructs 27 entries. Its `tools/evaluate_dynamics.py` and `tools/morphadapt_trace.py` bytes differ from the two recorded hashes. The recorded transient bytes are absent from commit history, unreachable Git objects, inspected server trees, the 2026-09-05 frozen tmp tree, and 488 inspected evidence ZIPs. Formal Table 2 evidence does not cite either transient hash.

Therefore:

```text
TRANSIENT_BINDING_HASH_BUG=YES
TABLE2_AUTHORITY_FOR_88FC_HASH=NO
TABLE2_AUTHORITY_FOR_76C70_HASH=NO
RECOVERABLE_TRANSIENT_BYTES=NO
```

The existing binding remains unchanged until behavior equivalence is demonstrated. In particular, its source-integrity gate remains fail-closed; the current arbitrary rmamorph checkout is not accepted as authority.

## Reproducible candidate

The candidate is the detached, clean Git commit `3893388b79b0df4b10ad3cd8504258618947325c`. `configs/modumorph_evaluator_behavior_regression.json` records its two actually recomputed normalized source hashes, the superseded transient hashes as non-authoritative history, and a `pending_behavior_regression` authority status. Combining those two candidate hashes with the unchanged 27 binding entries produces a source gate that passes the candidate commit and rejects the current differing rmamorph checkout.

Candidate authority requires a server replay against existing canonical evidence. A PASS produces `binding_repair_proposal.json`; it does not edit the binding on the server. The binding may be changed and committed in the ModuMorph development workspace only after that evidence is reviewed. The future binding must preserve the transient hashes under `superseded_transient_hashes`, mark them non-authoritative, point `source_reference_sha` to the reconstructible commit, and bind the reviewed behavior-regression evidence.

## Prepared server workflow

`scripts/run_modumorph_evaluator_behavior_regression_server.sh` creates or verifies the detached worktree under `./tmp/rmamorph_behavior_regression_3893388`, selects an eligible existing reference trace, replays exactly one existing Table 2 State+Action episode, compares it, and packages logs/results on success or failure. It never modifies or commits the rmamorph working tree, never generates the reference, and never launches ModuMorph training or the full Strict-OOD97 matrix.

Current status:

```text
FROZEN_EVALUATOR_BEHAVIOR_REGRESSION=NOT_RUN
BINDING_CONTRACT_REPAIRED=NO
SOURCE_INTEGRITY_GATE_PRESERVED=YES
MODUMORPH_245K_SMOKE_READY=NO
FORMAL_MODUMORPH_EVAL_READY=NO
```
