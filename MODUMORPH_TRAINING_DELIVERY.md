# Seed-1409 nominal ModuMorph throughput/pilot handoff

This delivery prepares server execution; Codex did not run MuJoCo, CUDA, benchmarks or training. The 10M reduced-budget pilot preserves the official100M learning-rate schedule prefix and completes122 native rollouts = **9,994,240 training transitions**. State-replay probes are excluded and rolled back. Changing NUM_ENVS changes sample/sampler semantics, so the pipeline holds32 lanes and exposes worker concurrency through native VECENV.IN_SERIES. Bare ft.yaml is not the official ModuMorph recipe; the checked-in entry applies the exact README Ours overrides.

Run from the server's current ModuMorph checkout after pulling this branch. The scripts change to their own repository root and verify the currently activated Python environment (printing executable/prefix and dependencies); they do not guess an environment name. Required existing native dependencies are listed in `docker/build_files/requirements.txt`; additional telemetry dependencies are `psutil` and `cloudpickle`. Missing dependencies fail before training and remain in the launch ZIP. Choose one physical GPU; the commands below use GPU2, which may be replaced. No multi-GPU learner or other seed is launched.

```bash
bash scripts/preflight_modumorph_training.sh 2
bash scripts/run_modumorph_training_server.sh 2 benchmark
bash scripts/run_modumorph_training_server.sh 2 pilot
```

The benchmark command runs the mandatory native16-worker default first, then CPU-affinity candidates1/4/8/32 as available (32envs for every trial). Each candidate trains3 complete rollouts, **245,760 transitions**, with a15-minute hard deadline covering the entire process. Own timed-out process groups are terminated; unrelated processes are never killed. The deadline does not assert that every machine can complete the benchmark in minutes. stdout/stderr are visible and saved. The coordinator records CPU utilization (100% = one logical core; summed learner+worker CPU time), whole process-tree peak RSS, selected-device average GPU utilization/peak memory and measured training transitions/second. GPU telemetry is device-wide and can include unrelated processes printed by preflight. The timer excludes state probes and final checkpoint/adapter acceptance; intermediate checkpoint cost is included in training wall time.

Candidates need zero non-finite/native numerical failures, completed process exit, original inference-loader acceptance, full-state replay equality, and numerical training-state digest equality with the default16-worker trial. Failed candidates have explicit failure JSON/logs. `selected.json` chooses the highest stable equivalent throughput; it may retain the default. No OMP/MKL/OpenBLAS overrides are made. Existing caller values and native torch1thread are recorded; changing thread settings requires a fresh benchmark.

The pilot reads `./tmp/modumorph_throughput_s1409/selected.json`, checks source/split/thread identity and trains only seed1409. It verifies the exact ordered100IDs and authority YAML hash, raw XML/metadata hashes and native loader acceptance by actually constructing all training environments before training. Native metadata/XML problems fail without conversion or skipping. Frozen assets are bound directly at:

`/home/yyf/Workspace/Code/rmamorph/output/unimals_100/train`

Expected final files:

- `./tmp/modumorph_pilot_s1409_10m/checkpoints/iteration_000122/Unimal-v0.pt`: original `[ActorCritic, ob_rms]` inference format, checked by a fresh original adapter worker.
- Same directory `resume.pt`: model state, Adam state, all RMS/return accumulator, next observation, sampler/meter/RNG/worker/simulator/wrapper state.
- Same directory `metadata.json`: actual transitions/iterations/optimizer updates, seed/count, Git/config/checkpoint hashes, `reduced_budget=true`, `formal_matched_baseline=false`.
- `configs/resolved.yaml`, `provenance/signature.json`, `logs/training.jsonl`, `logs/resources.json`, `status/summary.json`, `status/last_valid.json`.

Completed checkpoints are immutable. State is saved every10 completed iterations and on the last iteration; interrupted pending directories cannot replace the last valid pointer. Resume replays work after the last saved boundary; no weights-only resume or fresh environment reset is used. A new launch refuses an existing output directory. Resume command:

```bash
bash scripts/run_modumorph_training_server.sh 2 pilot --resume
```

Resume rejects changes to seed, resolved config (including workers/budget), ordered split/assets, architecture/source, optimizer/RMS contract, package versions, GPU model or thread environment. Last-valid metadata and checkpoint checksums must match. Native simulator state roundtrip/replay is required on fresh and resumed launches. This is locally tested with fake states; **real mujoco-py acceptance remains NOT_RUN**. If a resumed run is already at122 iterations, completion is recognized without further optimizer updates.

All scratch/results/logs and launch evidence ZIPs are under project `./tmp`. Launch ZIPs include compact text/JSON/YAML evidence on success or failure; large model/resume binaries remain in checkpoint directories. Interactive terminals remain open; `--no-keep-open` preserves the task exit code for automation.

Next action after a learned checkpoint is only a separately authorized1-walker real server smoke. No smoke, Strict-OOD97 gate or full evaluation runs automatically. The existing frozen evaluator source hash check currently fails against the adjacent dirty local rmamorph checkout; its frozen binding was preserved. Restore/use the certified evaluator checkout before later smoke, without overwriting parallel work.

## Validation and status boundaries

`MODUMORPH_TRAINING_AUDIT=PASS`, `LEGACY_TRAINING_PATH_IDENTIFIED=YES`, `PARALLELISM_IMPLEMENTED=YES`, `PARALLELISM_SEMANTIC_SAFE=YES` (fixed-lane config/source reasoning; real equivalence gate NOT_RUN), `ENV_STEP_ACCOUNTING=PASS`, `MORPHOLOGY_SPLIT_BINDING=PASS` (ordered authority binding; real assets NOT_RUN), `CHECKPOINT_SAVE_LOAD=PASS`, `RMS_SAVE_LOAD=PASS`, `RESUME_SAFETY=PASS` (local mock/contract tests only), `NO_DR_CONFIRMED=YES`, `REDUCED_BUDGET_PILOT=YES`, `SERVER_THROUGHPUT_BENCHMARK_REQUIRED=YES`, `SERVER_10M_PILOT_READY=NO`, `FORMAL_MATCHED_BASELINE=NO`.

Server readiness requires a successful actual benchmark, native state replay and checkpoint acceptance. No throughput improvement is claimed before this evidence exists. `SERVER_RUNTIME_VALIDATION=NOT_RUN`, `GPU_VALIDATION=NOT_RUN`, `FULL_EXPERIMENT=NOT_RUN`.

Local verification (already executed):

17 training checks passed, with no skips. Existing adapter checks:13 passed; the external frozen-source check failed for `tools/evaluate_dynamics.py` and `tools/morphadapt_trace.py` in the dirty adjacent rmamorph checkout. No frozen hash was updated to conceal this drift. A local launcher failure-path check also confirmed missing MuJoCo exits1 and emits a ZIP containing the original console traceback and exit code.

```powershell
python checks/modumorph_training_checks.py
python -m py_compile tools/modumorph_training_contract.py tools/run_modumorph_training.py metamorph/envs/vec_env/training_snapshot.py metamorph/algos/ppo/ppo.py metamorph/envs/vec_env/subproc_vec_env.py
```

Expected: unittest finishes `OK`; compile exits0. Shell launchers passed `bash -n`. Tests use MOCK weights/assets/simulation states only; actual original ActorCritic/RMS serialization and inference loader are exercised without MuJoCo.

## Files and GitHub Desktop handoff

Added: `MODUMORPH_TRAINING_AUDIT.md`, `MODUMORPH_TRAINING_SEMANTIC_FINGERPRINT.json`, `MODUMORPH_TRAINING_SEMANTIC_DIFF.md`, this handoff, `MODUMORPH_TRAINING_STATUS.json`, `configs/modumorph_training_binding.json`, `tools/modumorph_training_contract.py`, `tools/run_modumorph_training.py`, `metamorph/envs/vec_env/training_snapshot.py`, `scripts/preflight_modumorph_training.sh`, `scripts/run_modumorph_training_server.sh`, `checks/modumorph_training_checks.py`.

Changed: `metamorph/algos/ppo/ppo.py` (opt-in observer; disabled path regression checked) and `metamorph/envs/vec_env/subproc_vec_env.py` (opt-in training-state RPC only). No MorphAdapt source/evaluator or original configs/ft.yaml changed.

Summary: 增加保留原生语义的 ModuMorph 吞吐基准与10M试训入口

Description: 固定32个env和官方架构/PPO设置，使用IN_SERIES比较CPU worker吞吐；绑定冻结100个训练walker，增加实际transition计数、完整续训状态和显式reduced-budget provenance。17项本地检查和脚本语法检查通过；服务器训练未运行，冻结evaluator外部源码漂移保持显式阻塞。
