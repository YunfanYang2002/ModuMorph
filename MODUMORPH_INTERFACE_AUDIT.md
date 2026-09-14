# ModuMorph interface audit

Audit date: 2026-09-14. This is a source-level integration contract, not checkpoint, simulator, GPU, or experiment acceptance.

## Identity and policy boundary

README.md:1-5 identifies the ICML 2023 Universal Morphology Control via Contextual Modulation implementation built on MetaMorph. README.md:30-34 specifies the ModuMorph switches: fixed attention and hypernetwork enabled, linear context encoder, positional embedding disabled, embedding dropout disabled. `configs/ft.yaml` alone does not select ModuMorph: the saved resolved configuration is required.

The policy is a current-observation spatial transformer with morphology-conditioned attention and embedding/decoder weights. `metamorph/algos/ppo/model.py:128-219,240-359` processes limb tokens and context in the same call. No temporal history, previous action, recurrent hidden state, mutation label, adaptation latent, or privileged teacher context is an input to this implementation. Static nominal morphology may contain nominal mass and motor gear; it must never be replaced with current mutation values.

## Checkpoint and configuration

| Item | Source evidence | Integration requirement |
|---|---|---|
| Save format | `metamorph/algos/ppo/ppo.py:239-241` | Pickled list `[ActorCritic object, ob_rms]`, not a standalone state dictionary |
| Native restore | `metamorph/algos/ppo/inherit_weight.py:6-32` | Existing restore silently continues for absent names; adapter must reject missing/unexpected keys and shape discrepancies |
| Architecture | `metamorph/algos/ppo/model.py:62-219` | Use the checkpoint's exact resolved configuration; validate FA/HN flags and input widths |
| Padding override | `tools/train_ppo.py:17-44` | UNIMAL setup hardcodes MAX_JOINTS=16 and MAX_LIMBS=12, overriding defaults; do not call this setup on evaluator config |
| Batch size | `metamorph/algos/ppo/model.py:460-465,496-498` | Inference reshapes using `cfg.PPO.NUM_ENVS`; worker config must match actual inference batch |
| Standard deviation | `metamorph/algos/ppo/model.py:451-458,520-522` | Saved log_std controls distribution; deterministic action uses the mean |
| Determinism | `metamorph/algos/ppo/model.py:545-557`; `tools/evaluate.py:66-77` | Set worker DETERMINISTIC=True and actor eval mode; no-grad inference |
| Normalization | `metamorph/envs/vec_env/vec_normalize.py:23-34,92-115`; `metamorph/config.py:383-386` | Restore exact per-feature checkpoint mean/variance; epsilon=1e-8 and clip=10; freeze statistics |

Separate worker processes are justified because both repositories use the `metamorph` Python namespace and mutable global `cfg`. Do not import both packages into one evaluator process or overwrite MorphAdapt's protocol configuration. Reuse the evaluator's reset, mutation timing, seeds, episodes, aggregation and metrics unchanged; replace only the actor call through an explicit observation/action binding.

## Environment assumptions and unresolved evidence

Native environment uses gym 0.17.1 and mujoco_py 2.0.2.8 (`docker/build_files/requirements.txt:3,5`), torso/0 naming, one free root joint, at most x/y hinge slots per limb, and direct MuJoCo actuator control (`metamorph/envs/modules/agent.py:101-157,240-256`; `metamorph/envs/tasks/unimal.py:32,123-126,206-210`). These are schema assumptions, not automatically satisfied by another robot family.

No UNIMAL learned `.pt` checkpoint or its resolved `config.yaml` was found by the targeted local file inventory. Modular robot XML examples exist; their presence does not establish UNIMAL test assets. Server checkpoint, resolved configuration, checkpoint hashes, normalization payload, exact train/test assets and simulator compatibility are UNKNOWN until inspected on the actual server. MODEL_LOAD=NOT_RUN; SERVER_RUNTIME_VALIDATION=NOT_RUN; GPU_VALIDATION=NOT_RUN; FULL_EXPERIMENT=NOT_RUN.

Requested child routing is managed by the parent. This audit has no direct effective-model metadata and makes no effective-routing claim.
