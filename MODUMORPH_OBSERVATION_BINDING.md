# ModuMorph observation binding

This contract binds current state plus frozen nominal morphology to ModuMorph. No history, previous action, teacher privileged context or mutation parameters may cross the worker boundary. The checkpoint's resolved feature lists are authoritative; widths below describe repository defaults only.

## Default proprioception

`metamorph/config.py:358-365` and `D:/CODES/master/rmamorph/metamorph/config.py:383-388` declare the same default ordered feature list. Each limb row is limb features followed by two joint slots, with zero-filled absent joints (`metamorph/envs/modules/agent.py:265-300`). Default width is 30 + 2*11 = 52.

| Limb field | Width | Exact source/semantics |
|---|---:|---|
| body_xpos | 3 | data.body_xpos; subtract torso/0 x from x only, preserve y and z (`agent.py:217-222`) |
| body_xvelp | 3 | data.body_xvelp (`agent.py:224`) |
| body_xvelr | 3 | data.body_xvelr (`agent.py:225`) |
| body_xquat | 4 | data.body_xquat in native order (`agent.py:223`) |
| body_pos | 3 | nominal model.body_pos (`agent.py:228`) |
| body_ipos | 3 | nominal model.body_ipos (`agent.py:229`) |
| body_iquat | 4 | nominal model.body_iquat (`agent.py:230`) |
| geom_quat | 4 | nominal model.geom_quat at exact agent geometry indices (`agent.py:231`) |
| body_mass | 1 | frozen nominal mass; original reads live model mass (`agent.py:234`), rmamorph uses nominal (`rmamorph agent.py:168-169`) |
| body_shape | 2 | nominal geom_size[:, :2] (`agent.py:235`) |

| Joint field, repeated in x/y slots | Width per joint | Exact semantics (`agent.py:240-256`) |
|---|---:|---|
| qpos | 1 | `(qpos[7:] - range_min)/(range_max-range_min)`, no centering to [-1,1] |
| qvel | 1 | data.qvel[6:] |
| jnt_pos | 3 | nominal model.jnt_pos[1:] |
| joint_range | 2 | nominal model.jnt_range[1:] |
| joint_axis | 3 | nominal model.jnt_axis[1:] |
| gear | 1 | frozen nominal actuator_gear[:,0]; original live model read at :253; rmamorph nominal at :201-202 |

## Static context reconstruction

rmamorph's default observation has no ModuMorph `context`. Its `privileged_context` is a different teacher input and is forbidden as a substitute. Reconstruct context exactly from the nominal morphology model at simulator creation, before any mid-episode mutation, then cache it for that morphology. This matches ModuMorph `modify_sim_step` caching at `agent.py:128` and `observation_step` reuse at :305. An all-zero vector, raw unscaled proprioceptive subset, approximate XML parser, or currently mutated mass/gear is not an acceptable binding.

Default context row width is 17 + 2*9 = 35. Select fields in `CONTEXT_OBS_TYPES` order after applying native fixed scaling, then pack the same x/y joint slots. Full bounds are in `metamorph/envs/modules/agent.py:19-32`; extraction and scaling are :179-211. Every field uses `-(lower!=upper) + 2*(value-lower)/(upper-lower+1e-8)`, including equal-bound dimensions; there is no clipping.

| Context field | Width | Lower bound | Upper bound |
|---|---:|---|---|
| body_pos | 3 | [-0.5,-0.45,-0.49] | [0.5,0.45,2.04] |
| body_ipos | 3 | [-0.225,-0.225,-0.225] | [0.225,0.225,0] |
| body_iquat | 4 | [0.70710678,-0.70710678,-0.70710678,0] | [1,0.70710678,0.70710678,0] |
| geom_quat | 4 | same as body_iquat | same as body_iquat |
| body_mass | 1 | [1.17809725] | [4.1887902] |
| body_shape | 2 | [0.05,0] | [0.1,0.22627417] |
| jnt_pos (each joint slot) | 3 | [-0.05,-0.05,0] | [0.05,0.05,0.05] |
| joint_range (each slot) | 2 | [-1.57079633,0] | [0,1.57079633] |
| joint_axis (each slot) | 3 | [-0.5000024,-0.5000024,-1] | [1,1,1] |
| gear (each slot) | 1 | [0] | [300] |

The `BASE_CONTEXT_NORM='running'` config declaration does not establish a running-normalization implementation: inspected extraction always uses the fixed bounds above. Default VecNormalize normalizes only proprioceptive. If an actual checkpoint specifies additional normalized keys, inspect its complete RMS payload before accepting it.

## Packing and boundary validation

| Worker input | Required contract |
|---|---|
| proprioceptive | raw current/nominal rows, zero-padded to checkpoint MAX_LIMBS, flatten limb-major, then apply checkpoint RMS exactly once |
| context | scaled frozen nominal rows, zero-padded to checkpoint MAX_LIMBS, flatten limb-major; default no RMS |
| obs_padding_mask | False for real limbs, True for padding (`multi_env_wrapper.py:91-95`) |
| act_padding_mask | True for absent x/y joints and padded limbs, False for actuated slots (:103-115) |
| edges | native joint_to/joint_from flatten, padding checkpoint MAX_LIMBS-1 (:133-137; agent.py:159-177) |
| traversals/SWAT_RE | required only when saved config enables their respective switches (`model.py:481-487`); do not fabricate |
| hfield | only when checkpoint's ENV.KEYS_TO_KEEP requests it; use exact environment field encoder contract |

rmamorph appends task commands per limb when MOTION_COMMAND or VELOCITY_COMMAND is enabled (`rmamorph agent.py:173-185`). They occur after selected limb features and before joint slots. Never truncate the end of the flattened row to remove them. Require an explicit named raw-state binding; reject unmatched feature lists, command semantics, body/geometry order, padding capacity or RMS shape. A fixed forward-locomotion checkpoint cannot be relabeled a command-conditioned controller merely by dropping command fields.

Checkpoint-specific feature lists, normalization shape and nominal server assets remain UNKNOWN. Numeric native-vs-bound parity on real morphology observations is NOT_RUN and is required before scientific evaluation.

## Field-level acceptance ledger

All rows use original MuJoCo body order followed by x/y joint slots. Shape is per real limb; padded output is `[1,12*52]`. Proprioceptive normalization is checkpoint RMS, epsilon `1e-8`, clip `10`, applied once by the frozen evaluator. These statuses describe the source mapping; real checkpoint/XML binding remains UNKNOWN.

| source → destination | shape | units / coordinates | normalization | equivalence / status |
|---|---|---|---|---|
| frozen body_xpos → native body_xpos | 3 | metres; world y/z, torso-relative x; native height wrapper | RMS | same extraction, EXACT |
| frozen body_xvelp → native body_xvelp | 3 | m/s, MuJoCo world body linear velocity | RMS | same extraction, EXACT |
| frozen body_xvelr → native body_xvelr | 3 | rad/s, MuJoCo body angular velocity convention | RMS | same extraction, EXACT |
| frozen body_xquat → native body_xquat | 4 | wxyz world quaternion | RMS | same extraction, EXACT |
| frozen body_pos → native body_pos / context | 3 | metres, parent frame | RMS / native fixed bounds | EXACT / DERIVED_EQUIVALENT |
| frozen body_ipos → native body_ipos / context | 3 | metres, body inertial frame offset | RMS / fixed bounds | EXACT / DERIVED_EQUIVALENT |
| frozen body_iquat → native body_iquat / context | 4 | wxyz inertial orientation | RMS / fixed bounds | EXACT / DERIVED_EQUIVALENT |
| frozen geom_quat → native geom_quat / context | 4 | wxyz body-local geometry orientation | RMS / fixed bounds | EXACT / DERIVED_EQUIVALENT |
| immutable nominal body_mass → native body_mass / context | 1 | kg | RMS / fixed bounds | equal to native nominal training input; dynamic values deliberately unavailable, DERIVED_EQUIVALENT |
| frozen geom_size[:2] → native body_shape / context | 2 | metres, geometry size convention | RMS / fixed bounds | EXACT / DERIVED_EQUIVALENT |
| frozen normalized joint qpos → native qpos | 2*1 | fraction of each hinge range | RMS | same range mapping, EXACT |
| frozen joint qvel → native qvel | 2*1 | rad/s | RMS | same extraction, EXACT |
| frozen jnt_pos → native jnt_pos / context | 2*3 | metres, body-local joint anchor | RMS / fixed bounds | EXACT / DERIVED_EQUIVALENT |
| frozen jnt_range → native joint_range / context | 2*2 | radians | RMS / fixed bounds | EXACT / DERIVED_EQUIVALENT |
| frozen jnt_axis → native joint_axis / context | 2*3 | dimensionless body-local axis | RMS / fixed bounds | EXACT / DERIVED_EQUIVALENT |
| immutable nominal actuator gear → native gear / context | 2*1 | native MuJoCo transmission coefficient | RMS / fixed bounds | nominal training value, DERIVED_EQUIVALENT |
| contact state | — | no native default field | — | NOT_USED |
| goal / command | — | formal Table 2 profile has commands disabled | — | NOT_USED; command-enabled profile is rejected |
| real-limb prefix / missing x/y slots → native masks | 12 / 24 | True = invalid, limb-major | none | EXACT; runtime inventory UNKNOWN |
| edges → native edges | 2*16 | child/parent indices, padding index 11 | none | EXACT; runtime order UNKNOWN |
| static raw nominal fields → native context | 12*35 | field ordering and bounds above | fixed bounds before zero scatter | DERIVED_EQUIVALENT; server exact original comparison required |
| history / previous actions / privileged dynamics | — | never transported | — | NOT_USED |
| SWAT traversals / relation and external hfield | — | not in audited backend | — | NOT_USED; checkpoints requiring these are rejected |

The runner obtains double-precision raw values directly from the wrapped environment reset before DummyVecEnv buffers and normalization, then compares context with original `Agent.get_context` on the identical nominal model snapshot. It never reconstructs context from clipped tensors. The actual worker receives already normalized proprioception and the bound static context.
