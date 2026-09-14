# Formal runner, prepared only

No formal evaluation was started. Formal **rollout** is gated by a real COMPLETE smoke with identical checkpoint/config/backend/source/split hashes and asset roots. Smoke from another checkpoint or changed backend cannot approve this run. Passing dry-run only approves the asset preflight.

```bash
bash scripts/run_modumorph_eval_server.sh <CONDA_ENV> <PHYSICAL_GPU> \
  --morphadapt-root <MORPHADAPT_REPO> \
  --checkpoint <MODUMORPH_CHECKPOINT.pt> --config <RESOLVED_CONFIG.yaml> \
  --strict-identity <FROZEN_STRICT_OOD97_IDENTITY.tsv> \
  --walker-root <MORPHADAPT_TEST_ROOT> --train-xml-root <MODUMORPH_TRAIN_ROOT/xml> \
  --output ./tmp/modumorph_formal --mode formal \
  --smoke-evidence ./tmp/modumorph_real_smoke --dry-run
```

The command above is **preflight only**. After real smoke acceptance, remove --dry-run to execute the prepared 776-episode matrix (97 IDs × four native protocols × two settings × seed 1409). No new seeds, split, mutation settings or horizons are exposed as runtime tuning options. Add --resume only for an inspected identical interrupted run. Complete cells are hash-verified/skipped; unfinished cells retain prior attempts and rerun in new attempt folders. Native early-termination outcomes are retained, not silently excluded.

Consolidated raw JSON v3: raw/fixed_dynamics.json and raw/mutation.json. Per-step traces and event metrics reside in raw/<setting>_<protocol>_attempt_N/traces. Method identity is ModuMorph. No additional summary metric, fall metric or bootstrap is created. Use the existing Strict-OOD97 cell readers and current paper aggregation with separately supplied method/checkpoint provenance.

Checkpoint families/configs outside the audited UNIMAL 12-limb static fixed-attention/hypernetwork interface are rejected. Supporting classic Modular-v0, per-training-ID weights, extra hfield keys, SWAT/mirror, command-enabled observations, altered feature lists or additional RMS keys requires a new exact semantic binding; no approximate compatibility path is provided.
