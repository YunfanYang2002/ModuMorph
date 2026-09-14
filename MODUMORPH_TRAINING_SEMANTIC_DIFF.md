# ModuMorph accelerated reduced-budget pilot: semantic diff

Baseline is **ft.yaml + the official README Ours overrides**, not bare ft.yaml (which retains MetaMorph architecture defaults). The fingerprint was written at source commit `098b537337cee7a7c9dc0b6fef5b6260bfbf4996`, before training hooks changed. Model parameter count: 5,232,798, measured by constructing the original ActorCritic locally without a simulator.

| Item | Original official recipe | Pilot / accelerated candidates | Changed? |
|---|---|---|---|
| architecture | Original fixed-attention + hypernetwork Transformer, 5 layers, 128 embedding, 2 heads | Same native ActorCritic | NO |
| observation | Native current proprioceptive/model features + static morphology context; 52/35 features per padded limb | Same; no history, previous action or privileged dynamics | NO |
| action | Native sampled Gaussian, fixed std 0.9, node padding and motor mapping | Same | NO |
| reward | Native nominal locomotion + wrappers + return normalization | Same | NO |
| PPO loss | Native clipped policy and value objectives, KL early stop coefficient 5 | Reuses original train_on_batch | NO |
| gamma | 0.99 | 0.99 | NO |
| GAE | 0.95, native done/timeout handling | Same | NO |
| optimizer | Adam, EPS 1e-5, weight decay 0, gradient clip 0.5 | Same; actual optimizer steps counted | NO |
| LR | 3e-4 cosine, native 5-iteration warmup, original MAX_ITERS=1220 denominator | Same schedule prefix; early exit does not compress cosine into 122 iterations | NO |
| rollout horizon | 2560 steps per env lane | 2560 | NO |
| rollout/minibatch/epochs | 81,920 / 5120 / 8 | Same; at most 128 updates per rollout, KL can stop early | NO |
| morphology sampler | balanced_replay_buffer; lane chunks, EMA after iteration 30 | Same implementation and lane ordering | NO |
| train split | Native root default with unspecified os.listdir order | Exact ordered 100-ID frozen MorphAdapt training authority, hash checked; no walker generation/conversion | YES: explicitly required binding |
| num envs | 32 | 32; --num-envs rejects other values | NO |
| CPU workers | 16, IN_SERIES=2 | Candidates 1/4/8/16/32, IN_SERIES=32/workers, filtered by CPU affinity plus mandatory default16 | Possibly YES |
| total env-step budget | 100M requested, 99,942,400 complete-rollout transitions | 10M requested, **9,994,240** complete-rollout transitions (122 iterations; remainder5760) | YES: reduced budget |
| simulator/reset/physics | Agent/Floor, NEW_SIM_ON_RESET=True, native reset noise/termination | Same; no DR | NO |
| checkpoint I/O | cur_iter%100 plus final; model + observation RMS only | Every10 completed iterations plus final; immutable completed checkpoint and full resume state | YES: engineering |
| logging | Native training meter and TensorBoard; native intermediate FPS uses final iteration and can overstate | Native logs retained; additional measured transition throughput/resource telemetry | YES: engineering |
| CPU threads | Native learner torch.set_num_threads(1), caller BLAS environment | Same; no automatic OMP/MKL/OpenBLAS changes; values recorded and checked | NO |

`PARALLELISM_SEMANTIC_SAFE=YES` is a source/config assertion for fixed-lane worker grouping; server equivalence is **NOT_RUN**. Benchmark selection additionally requires exact numerical model/RMS/simulator/sampler/meter digest equality to the default16-worker run, zero non-finite/native numerical errors, successful complete process exit within15minutes per candidate, and original inference checkpoint acceptance. Candidates failing equivalence are excluded, not relabeled safe. The coordinator reports every failed candidate separately.

The opt-in observer leaves the legacy no-observer rollout, PPO math, sampler, LR and model-save branch intact. Training state restores Adam, observation and return RMS, per-lane return accumulator, current observation, sampler file, meter histories, host/worker/env RNG and nominal simulator/wrapper state. Binary MJB models avoid XML serialization rounding. This follows the documented [mujoco-py model/state serialization API](https://github.com/openai/mujoco-py/blob/master/mujoco_py/mjsim.pyx); simulator integration/warm-start state is saved separately. A real16-step per-lane save/restore/replay equality gate is mandatory before any benchmark/pilot training. Local fake-state tests cannot certify this gate.

No learned checkpoint was produced by Codex. `SERVER_RUNTIME_VALIDATION=NOT_RUN`, `GPU_VALIDATION=NOT_RUN`, `FULL_EXPERIMENT=NOT_RUN`, `REDUCED_BUDGET_PILOT=YES`, `FORMAL_MATCHED_BASELINE=NO`. Pilot completion permits only the separately requested1-walker server smoke as the next action; neither smoke nor evaluation runs automatically.
