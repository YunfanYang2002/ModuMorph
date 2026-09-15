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

The binding now uses the two normalized hashes from the reconstructible source at `3893388b79b0df4b10ad3cd8504258618947325c`. The server behavior regression passed against the canonical Table 2 raw result with exact raw output (`max_abs_diff=0.0`, `max_rel_diff=0.0`). The source-integrity gate remains fail-closed; the clean `3893388` source passes while the current differing rmamorph checkout is rejected.

## Reproducible candidate

The approved source is the detached, clean Git commit `3893388b79b0df4b10ad3cd8504258618947325c`. `configs/modumorph_frozen_eval_binding.json` records its two recomputed normalized source hashes as active authority. It also preserves the superseded transient hashes, their `not_recoverable` status, and `table2_authority_for_transient_hashes=false` as historical provenance.

The repair authority is the server-generated `./tmp/modumorph_evaluator_behavior_regression/binding_repair_proposal.json`. The binding records that artifact and its report identity together with the reviewed acceptance fields. No unavailable server checksum was invented in the local repository.

## Completed behavior regression

The server selected an existing canonical Table 2 raw result, replayed the same one-walker mutation cell from the clean `3893388` source, confirmed mutation at step 250, and passed raw structure, discrete, and continuous comparison. The server evidence remains under `./tmp/modumorph_evaluator_behavior_regression/`.

Current status:

```text
FROZEN_EVALUATOR_BEHAVIOR_REGRESSION=PASS
BINDING_CONTRACT_REPAIRED=YES
SOURCE_INTEGRITY_GATE_PRESERVED=YES
MODUMORPH_245K_SMOKE_READY=YES
FORMAL_MODUMORPH_EVAL_READY=NO
```
