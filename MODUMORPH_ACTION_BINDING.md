# ModuMorph action binding

## Policy output

For UNIMAL, `ActorCritic` produces two scalar action means per padded limb and `2*MAX_LIMBS` total actions (`metamorph/algos/ppo/model.py:435-440`). `Agent.act` returns `pi.loc` when `cfg.DETERMINISTIC=True`; otherwise it samples (`model.py:545-553`). Evaluation requires deterministic mean, eval mode and no-grad. Actor means are not multiplied by gear, clipped, squashed through tanh, or rescaled by this source path.

| Stage | Source | Binding |
|---|---|---|
| Limb slot order | `metamorph/envs/modules/agent.py:130-157` | torso slots absent; each limb has x then y hinge positions determined from joint names |
| Output flatten | `metamorph/algos/ppo/model.py:351-367` | limb-major output, two slots per limb |
| Padding mask | `metamorph/envs/wrappers/multi_env_wrapper.py:103-115` | True means absent or padded; false means actual actuator slot |
| Mirroring | `multi_env_wrapper.py:173-177`; `agent.py:298-299` | If enabled, inverse permutation must match native metadata; reject unsupported mirroring rather than ignore it |
| Actuator extraction | `multi_env_wrapper.py:179-180` | `padded_action[~act_padding_mask]`, exactly once, preserving simulator actuator order |
| Actuator limits | `metamorph/envs/tasks/unimal.py:123-126` | simulator actuator_ctrlrange defines native bounds |
| Control application | `unimal.py:206-210` | write action directly into sim.data.ctrl and run frame_skip=4 (:32) |

The worker should return a padded deterministic action to MorphAdapt's unchanged native action wrapper only when its mask, slot order and padding width exactly match. Otherwise perform the explicit inverse slot-to-actuator mapping at the boundary and return actual actuator controls through a clearly defined evaluator hook. Never unpad twice or feed a 24-slot checkpoint vector into a differently sized wrapper. Do not alter the evaluation simulator's gear, timestep, control costs, reset, termination or mutation behavior to accommodate a policy.

## Acceptance checks

Reject mismatched actuator count/order, non-finite means, unsupported joint names, insufficient checkpoint padding, and incompatible action dimensions. Verify native ModuMorph inference versus bound inference on the same raw state/nominal model: normalized input arrays, masks, context and deterministic padded means must agree numerically; extracted actuator vectors must have exact order and count. Confirm zero mutation-induced changes to cached context and nominal morphology features while current proprioceptive state evolves.

Native clipping behavior downstream in the server MuJoCo version, real actuator limits, checkpoint mean parity and actual action mapping have not been executed. SERVER_RUNTIME_VALIDATION=NOT_RUN; GPU_VALIDATION=NOT_RUN; FULL_EXPERIMENT=NOT_RUN. These documents do not certify scientific comparability or completed evaluation.

The implemented adapter forwards the raw Gaussian mean without unpadding, clipping, tanh, smoothing, torque conversion, gear multiplication or motor-strength multiplication. Frozen `MultiUnimalNodeCentricAction` performs the sole valid-slot extraction; frozen `UnimalEnv.do_simulation` writes it unchanged into `sim.data.ctrl`. MuJoCo applies actuator limits (`assets/unimal.xml` motor defaults: ctrlrange [-1,1], ctrllimited=true). Invalid output slots can contain arbitrary means because the native environment discards them; zeroing these in the adapter would change the policy-side trace. The clipped executed-command history contract is distinct from the raw policy-output trace contract.

Source-level action dimension and flattening are EXACT. Server binding additionally requires `[1,24]`, 12 limbs, binary masks with non-actuated torso/padding, valid slot count equal to `model.nu`, and `model.actuator_trnid[:,0]` matching `model.joint_names[1:]`. Actual XML ordering/limits and mean parity remain UNKNOWN until real smoke. No actuator contract is approximated for incompatible checkpoints.
