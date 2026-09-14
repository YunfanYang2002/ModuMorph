# Real server smoke

Codex did not access a controlled Linux server or launch a rollout. Required: the ModuMorph fork containing this delivery, matching hash-bound MorphAdapt sources, original learned UNIMAL FA/HN checkpoint, its **resolved training config.yaml**, frozen Strict-OOD97 identity TSV, exact XML+metadata and complete ModuMorph train XML inventory.

From the Linux ModuMorph repository, substitute actual paths/environment/GPU:

```bash
bash scripts/run_modumorph_eval_server.sh <CONDA_ENV> <PHYSICAL_GPU> \
  --morphadapt-root <MORPHADAPT_REPO> \
  --checkpoint <MODUMORPH_CHECKPOINT.pt> --config <RESOLVED_CONFIG.yaml> \
  --strict-identity <FROZEN_STRICT_OOD97_IDENTITY.tsv> \
  --walker-root <MORPHADAPT_TEST_ROOT> --train-xml-root <MODUMORPH_TRAIN_ROOT/xml> \
  --output ./tmp/modumorph_real_smoke --mode smoke
```

Default walker is the first frozen identity row (override --walker only with a Strict-OOD97 ID); seed is 1409, protocol ood_strong, mutation at 250, one episode, existing 1000-step horizon. GPU is selected through CUDA_VISIBLE_DEVICES; --device defaults to logical cuda:0. Environment is explicitly activated; missing dependencies fail. All temporary paths, logs and archives are under project ./tmp.

For asset-only preflight, add **--dry-run** and choose a fresh ./tmp output directory. This performs no model loading or rollout and cannot certify inference. Expected success: ASSET_PREFLIGHT=PASS, no errors, ROLLOUT_LAUNCHED=NO. Missing assets return exit 1 and list exact missing items.

Real smoke success requires status.txt=COMPLETE and valid COMPLETE.json hashes, successful original checkpoint/config validation, exact original-vs-bound nominal proprioception/context, actuator count/order, finite forward/action/trace, actual mutation step=250, completed episode and native metrics. Early termination before mutation **fails smoke certification**, even though such an episode remains a valid outcome in the formal protocol. Null unrecovered times remain valid; no NaN/Inf is allowed. Both outcomes print RUN_EXIT_CODE and OUTPUT_ZIP. Send the ZIP rather than large raw logs.

Launcher retains an interactive shell on success/failure; type exit to close it. For automation, insert --no-keep-open immediately after the GPU argument; the original failure exit code is preserved. Resume is explicit --resume; never delete prior results to make the command succeed.

SERVER_RUNTIME_VALIDATION=NOT_RUN; GPU_VALIDATION=NOT_RUN; SERVER_SMOKE_REQUIRED=YES; FORMAL_MODUMORPH_EVAL_READY=NO.
