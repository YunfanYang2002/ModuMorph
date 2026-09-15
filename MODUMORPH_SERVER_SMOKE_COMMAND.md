# 245K evaluator server smoke

The frozen evaluator behavior regression passed and the binding now authorizes the clean detached `3893388` source. The 245,760-transition checkpoint is eligible for one-walker smoke only; it is not performance evidence.

From the Linux ModuMorph repository, run:

```bash
cd ~/Workspace/Code/ModuMorph
git pull --ff-only

bash scripts/run_modumorph_eval_server.sh rmamorph 2 --no-keep-open \
  --morphadapt-root /home/yyf/Workspace/Code/ModuMorph/tmp/rmamorph_behavior_regression_3893388 \
  --checkpoint /home/yyf/Workspace/Code/ModuMorph/tmp/modumorph_throughput_s1409/workers_32/checkpoints/iteration_000003/Unimal-v0.pt \
  --config /home/yyf/Workspace/Code/ModuMorph/tmp/modumorph_throughput_s1409/workers_32/configs/resolved.yaml \
  --strict-identity /home/yyf/Workspace/Code/rmamorph/tmp/morphadapt_canonical_student_formal_table2_20260905T111437Z/manifests/strict_ood97_identity.tsv \
  --walker-root /home/yyf/Workspace/Code/rmamorph/output/unimals_100/test \
  --train-xml-root /home/yyf/Workspace/Code/rmamorph/output/unimals_100/train/xml \
  --output ./tmp/modumorph_245k_smoke_s1409 --mode smoke
```

Default walker is the first frozen identity row (override --walker only with a Strict-OOD97 ID); seed is 1409, protocol ood_strong, mutation at 250, one episode, existing 1000-step horizon. GPU is selected through CUDA_VISIBLE_DEVICES; --device defaults to logical cuda:0. Environment is explicitly activated; missing dependencies fail. All temporary paths, logs and archives are under project ./tmp.

For asset-only preflight, add **--dry-run** and choose a fresh ./tmp output directory. This performs no model loading or rollout and cannot certify inference. Expected success: ASSET_PREFLIGHT=PASS, no errors, ROLLOUT_LAUNCHED=NO. Missing assets return exit 1 and list exact missing items.

Real smoke success requires status.txt=COMPLETE and valid COMPLETE.json hashes, successful original checkpoint/config validation, exact original-vs-bound nominal proprioception/context, actuator count/order, finite forward/action/trace, actual mutation step=250, completed episode and native metrics. Early termination before mutation **fails smoke certification**, even though such an episode remains a valid outcome in the formal protocol. Null unrecovered times remain valid; no NaN/Inf is allowed. Both outcomes print RUN_EXIT_CODE and OUTPUT_ZIP. Send the ZIP rather than large raw logs.

Launcher retains an interactive shell on success/failure; type exit to close it. For automation, insert --no-keep-open immediately after the GPU argument; the original failure exit code is preserved. Resume is explicit --resume; never delete prior results to make the command succeed.

SERVER_RUNTIME_VALIDATION=NOT_RUN; GPU_VALIDATION=NOT_RUN; MODUMORPH_245K_SMOKE_READY=YES; FORMAL_MODUMORPH_EVAL_READY=NO.
