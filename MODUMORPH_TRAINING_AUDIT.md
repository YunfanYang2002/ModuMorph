# ModuMorph 原生训练审计

审计日期：2026-09-14。范围：本地源码、README 配方与 MorphAdapt 冻结训练配置；未运行训练、MuJoCo、GPU 或服务器。以下结论区分源码可确定的行为与需要实测的吞吐、轨迹等价性。

## 用户 20 个问题的逐项结论

| 问题 | 原生行为及证据 |
|---|---|
| 1. vectorized env | 已有，PPO初始化调用make_vec_envs，返回SubprocVecEnv→VecNormalize→VecPyTorch；`ppo.py:28–30`; `envs.py:98–116`。 |
| 2. environment数量 | `PPO.NUM_ENVS`，默认32；FIX_ENV/Modular多形态分支可能覆盖，见第5/14节。 |
| 3. CPU worker数量 | `VECENV.IN_SERIES`间接控制，P/series；默认16 workers，见第12节。 |
| 4. rollout length | 每lane2560步，见第5节。 |
| 5. batch size | 每rollout81,920 transitions，见第6节。 |
| 6. minibatch size | `PPO.BATCH_SIZE=5120`，见第7节。 |
| 7. PPO epochs | 8，受KL early stop影响，见第7/8节。 |
| 8. optimizer update对应transitions | 每次Adam.step使用5120个已收集样本；不对应5120个新env steps。同一81,920 rollout最多被8个epoch复用、128次Adam.step，见第7/8节。 |
| 9. termination | 迭代上限MAX_ITERS由总env-step预算整除P×T推导；100M→1220 rollouts，10M earlyexit→122 rollouts，见第9/11节。无逐步精确10M截断。 |
| 10. checkpoint interval | 每cur_iter%100==0，即首次rollout结束cur_iter0也保存；结束再save_model(-1)，见第19节。 |
| 11. checkpoint naming | `OUT_DIR/Unimal-v0.pt`持续覆盖，同时`checkpoint_<cur_iter>.pt`，结束为`checkpoint_-1.pt`；`ppo.py:236–241`。 |
| 12. resume | 原生加载是参数迁移/finetune，不是完整续训，见第19/20节。 |
| 13. normalization/RMS | checkpoint仅保存ob_rms，未保存reward ret_rms/每lane ret，见第19/20节。 |
| 14. model config保存 | CLI合并、set_cfg_options后写`OUT_DIR/config.yaml`，`train_ppo.py:219–236`; `config.py:559–565`。checkpoint另含序列化model对象；config本身不是完整resume provenance。 |
| 15. morphology sampler | learner写sampling.json、env按lane读取chunk，balanced replay依赖meter，见第16节。 |
| 16. train walker root | 原生默认`./unimals_100/train`；冻结权威为rmamorph resolved配置中的`./output/unimals_100/train`及100 IDs，见第3/4节。 |
| 17. RNG propagation | 父进程全局seed、每lane seed+idx、env私有RNG，见第15节。 |
| 18. GPU usage | 默认`DEVICE=cuda:0`（config521行）；model及obs送此device（ppo33/48行；pytorch_vec_env.py40–45行），actions回CPU NumPy（22行），MuJoCo在CPU env worker。GPU耗时/占用未测。 |
| 19. multi-GPU | 已审计的原生train_ppo/PPO路径没有DDP/DataParallel/torchrun或多learner调度；只有一个cfg.DEVICE。配置注释提DDP不是此入口已实现多GPU的证据。不能以多CPU workers推出多GPU支持。 |
| 20. 安全增大NUM_ENVS | 不能声称保持每update sample semantics；dataset、更新节奏、sampler、预算/LR与lane RNG都变。优先仅调IN_SERIES；静态安全候选但运行等价NOT_RUN，见第13/14节。 |

表中的简称源码路径均相对本repo：`metamorph/algos/ppo/ppo.py`、`metamorph/algos/ppo/envs.py`、`metamorph/config.py`、`metamorph/envs/vec_env/pytorch_vec_env.py`。

## 1. 原生训练入口与执行拓扑

入口为 `tools/train_ppo.py`，先合并 YAML 与命令行配置、推导配置、保存 resolved config，再启动 PPO（215–237 行）。`ppo_train()` 设置种子、`torch.set_num_threads(1)` 并实例化单个 PPO learner（199–208 行）。环境 subprocess 不是多个 PPO/DDP learner；不应把 CPU worker 数当作 WORLD_SIZE。

## 2. bare ft 是否是官方 ModuMorph

不是。`configs/ft.yaml` 仅指定种子 1409、Agent/Floor 模块与 terrain size。README 的 “Ours (ModuMorph)” 还指定 `PPO.KL_TARGET_COEF=5.`、`POS_EMBEDDING=None`、`EMBEDDING_DROPOUT=False`、`FIX_ATTENTION=True`、`HYPERNET=True`、`CONTEXT_ENCODER=linear`。bare ft 继承默认 KL20、learned positional embedding、embedding dropout 开启、FA/HN 关闭（`metamorph/config.py:263,433–454`），对应 README 的 MetaMorph 配方。科学比较必须固定官方 Ours 配方与 resolved config。

## 3. 固定训练形态权威

权威为 `D:/CODES/master/rmamorph/configs/morphadapt_metamorph_dr_matched_s1409_reference.yaml`：124–223 行列出有序 100 个训练 walker IDs；224 行路径为 `./output/unimals_100/train`。其 347/349/356 行指定 NUM_ENVS32、TIMESTEPS2560、seed1409；113 行指定 balanced replay sampling。不要导入整份 rmamorph YAML：其中含 ModuMorph 原生配置不支持的扩展字段，形态权威与 ModuMorph 方法配置应分别绑定。

## 4. 形态 ID 推导与顺序风险

`tools/train_ppo.py:94–105` 在 WALKERS 为空时从 XML 目录的未排序 `os.listdir` 推导 ID。ModuMorph 默认目录为 `./unimals_100/train`，本地该 XML 目录不存在。必须显式提供权威的完整 ID 顺序和相应 XML/metadata 文件；仅验证 ID 集合相等不足以保持形态索引与采样序列。

## 5. 原生环境数与 horizon

原生默认 `PPO.NUM_ENVS=32`、`PPO.TIMESTEPS=2560`（`metamorph/config.py:271–275`）。它们是逻辑环境 lane 数与每 lane 每 rollout 的步数；不等于 CPU process 数。

## 6. 每 rollout 样本数

每 rollout 为 `32 × 2560 = 81,920` env transitions。buffer 的训练 dataset 大小直接为该乘积（`metamorph/algos/ppo/buffer.py:91–100`）。环境步计数为 `(cur_iter+1) × NUM_ENVS × TIMESTEPS`（`ppo.py:259`）；不是 MuJoCo 内部 frame 数。

## 7. minibatch、epoch 与 optimizer 更新

默认 minibatch5120、epochs8（`config.py:252–256`），因此每 epoch16 minibatches，每 rollout 最多128次 Adam.step。`buffer.py:96–100` 的 sampler 使用 `drop_last=True`；本配置恰好整除。实际 Adam 次数可能因 KL early stop 减少。

## 8. KL early stop

`ppo.py:171–177` 在 minibatch backward/Adam 之前判断 `approx_kl > KL_TARGET_COEF × 0.01`，触发即退出整个本次 rollout 的更新。官方 Ours 阈值为0.05，bare ft 为0.20。不要将最多128次更新表述为实际已执行128次。

## 9. 原生 100M 预算

`tools/train_ppo.py:84–90` 以整数除法推导 MAX_ITERS。100,000,000预算得到1220 rollouts，实际99,942,400 transitions。预算是上限式整 rollout 截断；不会恰好达到100M。

## 10. 学习率与 warmup

默认 cosine、BASE_LR3e-4、MIN_LR0、WARMUP_ITERS5、WARMUP_FACTOR0.1（`config.py:278–288`）。`metamorph/utils/optimizer.py:7–10,32–40` 用 `cur_iter/MAX_ITERS` 计算 cosine，并在最初5次迭代乘线性 warmup；`ppo.py:92–93` 每 rollout 更新 LR。MAX_ITERS 是训练 schedule 语义。

## 11. 10M prefix 的正确预算

10,000,000目标按原生完整 rollout 截断为122 rollouts，即9,994,240 transitions。使用 EARLY_EXIT_STATE_ACTION_PAIRS10M 和 EARLY_EXIT=True，同时保持 MAX_STATE_ACTION_PAIRS100M、MAX_ITERS1220，可在进入 cur_iter122 前退出（`train_ppo.py:84–90`; `ppo.py:87–93`）。这保持100M计划的前122次 LR；将总预算直接改10M会把 cosine horizon 改为122，形成另一种训练协议。

## 12. 现有 CPU worker knob

现有参数是 `VECENV.IN_SERIES`，默认2，TYPE为 SubprocVecEnv（`config.py:486–493`）。`subproc_vec_env.py:60–76` 要求 env 数可整除 IN_SERIES，进程数为 `NUM_ENVS // IN_SERIES`。P32下，series2/1/4分别为16/32/8 CPU workers，无需新增 worker 框架。

## 13. 固定 PPO 形状时改变 worker grouping

静态源码支持只改 IN_SERIES、保持 P32/T2560/minibatch5120/epochs8/方法配置/种子与形态权威固定。它仅改变连续 env lanes 的进程分组；`np.array_split` 保持 lane 顺序，actions/results/get_unimal_idx 均按原顺序分组和展平（`subproc_vec_env.py:75–76,102–115,152–175`）。这是可用的工程优化候选，尚未证明吞吐提升或轨迹精确等价：`WORKER_GROUPING_RUNTIME_EQUIVALENCE=NOT_RUN`。

## 14. 改 NUM_ENVS 的影响

即使 horizon、minibatch固定，改 P 也改变 rollout dataset、每 epoch minibatch数、单位样本的 policy-update频率、预算整除和 MAX_ITERS/LR进度、env seed集合、形态列表长度及每 lane的形态 chunk。因此属于实验协议变化，不能作为等价 CPU worker 优化。`envs.py:83–96` 另有 FIX_ENV/Modular 路径会将 P 改成两倍形态数；必须检查实际 resolved/runtime P。

## 15. lane、worker 与 RNG 映射

`envs.py:70–82` 依次创建 env_idx0..P-1，`:31` 使用 `seed+env_idx`；MultiEnvWrapper 保留 env_idx（`multi_env_wrapper.py:19–25`）。改变 grouping 不改变这个 seed映射。全局 Python/NumPy/Torch RNG由 `sample.py:116–122` 设置；Gym env有自己的 NumPy RNG（`unimal.py:129–133`），模块使用该 RNG（`:74–80`），reset noise及mirroring也用它（`:105–113,169–182`）。这不保证 GPU算子的 bitwise determinism。

## 16. morphology sampler

`ppo.py:318–361` 在 learner进程生成 sampling.json：balanced策略前30次使用uniform episode长度，此后用per-agent episode长度统计推导概率，并由全局 `np.random.choice` 采样。列表长度依赖 P、T和平均episode长度。`multi_env_wrapper.py:55–59` 按P切列表并按原env_idx取chunk；episode reset递增序列索引（31–40行），每T步读取sampling.json更新序列（44–50行）。形态 sampler状态与meter历史都是继续训练状态。

## 17. 原生同步与串行瓶颈

每步依次执行 learner action inference、全体env step、buffer insert（`ppo.py:95–120`），没有learner/collector流水线。每worker内部serial step其env组（`subproc_vec_env.py:22–25`），父进程recv全部worker形成barrier（109–115行）。若 PER_NODE_EMBED开启，每步还有get_unimal_idx RPC（`ppo.py:97–98`）。更多worker是否有效需要测量IPC与最慢env/reset耗时。

## 18. sim重建与运行边界

`unimal.py:88–103,152–156` 在NEW_SIM_ON_RESET开启时重建MuJoCo sim；冻结权威明确该值为True（reference104行）。不要为了吞吐改为False，否则改变reset/dynamics生成行为。原生 `envs.py:103` 强制 multiprocessing fork，属于Linux/fork运行路径，不能从本地Windows静态审计声称服务器或MuJoCo可运行。

## 19. 原生 checkpoint 与 resume

`ppo.py:236–241` 仅保存 `[actor_critic, ob_rms]`；每100个cur_iter保存（152–153行），入口结束时save_model(-1)（`train_ppo.py:211`）。这是推理/参数迁移artifact，不是完整训练resume。`inherit_weight.py:6–45` 加载并复制参数，缺失名称时打印后继续，可能按FINETUNE配置冻结参数；不会恢复optimizer、迭代、RNG、meter、环境状态。原生train再次生成sampling并reset env（`ppo.py:70–72`），故不能宣称10M→100M精确续训。

## 20. 安全 continuation 所需状态与验收

完整resume至少需要optimizer及schedule/iteration/env-step状态、所有父进程RNG、per-agent meter采样历史、sampling.json、每lane私有RNG、active morphology、wrapper序列/索引/num_steps、MuJoCo sim及模块状态、episode/TimeLimit/统计计数、当前next observation、VecNormalize ob_rms/ret_rms/ret（`vec_normalize.py:30–34`）。原生worker RPC没有state save/restore命令（`subproc_vec_env.py:20–46`）。新resume机制必须严格验证源码/config/形态asset provenance，并以不中断run对比save/reload续段；只有权重加载不足以验收。

当前状态：`SOURCE_AUDIT=PASS`；`LOCAL_TRAINING=NOT_RUN`；`SERVER_RUNTIME_VALIDATION=NOT_RUN`；`GPU_VALIDATION=NOT_RUN`；`WORKER_GROUPING_RUNTIME_EQUIVALENCE=NOT_RUN`；`FULL_EXPERIMENT=NOT_RUN`。后续应先做小规模受控pilot，记录实际P/T/worker数、rollout样本数、Adam计数、LR、阶段耗时及failure evidence；不以pilot等同科学训练完成。
