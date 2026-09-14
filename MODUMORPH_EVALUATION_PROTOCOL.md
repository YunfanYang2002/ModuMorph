# Frozen evaluation binding

## Current authority

The current Table 2 pipeline is **Strict-OOD97**, 97 walker IDs / 87 exact-XML clusters, not the historical OOD98 campaign. Evidence inspected: MorphAdapt `tools/reaggregate_morphadapt_strict_ood97.py` (`frozen_ood97` removes the documented exact-training-XML collision), `tools/prepare_morphadapt_canonical_student_formal_table2.py`, and current `tmp/morphadapt_table2_paper_integration_staging/TABLE2_MANUSCRIPT_INTEGRATION_REPORT.md`. The locally present canonical preparation summary is a failed missing-assets preflight, not proof of a completed canonical evaluation.

`configs/modumorph_strict_ood97.txt` preserves OOD98 manifest ordering with only the documented excluded ID removed. Real runs also require the authoritative **frozen identity TSV**, not this name list alone. Preflight checks all 97 IDs, 87 actual XML hash clusters, identity hashes, cluster sizes and zero XML/ID overlap with the complete training inventory named by the ModuMorph resolved config. No local XML/identity evidence was available; split runtime binding is UNKNOWN.

Table 2 canonical renderer and the P1 formal continuation both explicitly use evaluation seed **1409**. The older campaign's five evaluation seeds describe a different campaign. Training seeds are not evaluation seeds; no ModuMorph training seed is inferred from a folder name. This delivery runs `[1409]` paired by walker / protocol / setting, without claiming paired training seeds.

## Reused entry points and boundaries

MorphAdapt root inspected: `D:/CODES/master/rmamorph`, HEAD `3893388b79b0df4b10ad3cd8504258618947325c`. Its unrelated nonstationary working-tree edits were preserved. None of its files were modified.

| Item | Authoritative implementation |
|---|---|
| Formal inference / rollout / aggregation | `tools/evaluate_dynamics.py`: `evaluate_protocol`, `evaluate_walker` |
| Deterministic inference | `tools/evaluate_v1.py`: `select_action` takes distribution mean |
| Environment factory / seeds | `metamorph/algos/ppo/envs.py`: make_vec_envs / make_env; RNG_SEED+rank; evaluator reseeds per walker/seed |
| XML resolution / module construction | `metamorph/envs/tasks/task.py`, `metamorph/envs/modules/agent.py`: walker-root/xml/ID.xml + canonical base XML |
| Metadata | `metamorph/envs/tasks/unimal.py`: walker-root/metadata/ID.json |
| Static morphology / default observation | `metamorph/envs/modules/agent.py`: nominal hardware fields; 52 per limb |
| Dynamics draw and application | `metamorph/envs/dynamics.py`: sample_parameters, capture_nominal_dynamics, apply_parameters, maybe_apply_mid_episode_perturbation |
| Mutation boundary | `metamorph/envs/tasks/unimal.py`: increment step_count, apply mutation, then physics stepping |
| Commands | frozen reference YAML disables velocity/motion commands; evaluator signal is x_vel, not command tracking |
| Termination | native gym 1000-step TimeLimit and TerminateOnFalling; stand-height and other native wrappers unchanged |
| Normalization | frozen VecNormalize with **ModuMorph checkpoint** RMS; training=False, epsilon=1e-8, clip=10; no reward normalization |
| Action | native two-slot node wrapper → valid-slot extraction → sim.data.ctrl; MuJoCo ctrlrange; no backend scaling |
| Trace | native EpisodeTraceWriter, schema version retained, no removed step fields |
| MetaMorph-DR backend | same evaluator loads serialized ActorCritic, ADAPT.ENABLED=false; no separate dynamics evaluator |
| Table metrics / clustered statistics | strict reaggregation's metrics_from_cell and existing table closure consumers; no new metric or bootstrap |

28 relevant evaluator/environment/source inputs and the resolved MetaMorph-DR reference environment YAML are hash-bound in `configs/modumorph_frozen_eval_binding.json`. A changed source/config fails preflight. Original ModuMorph source/config/adapter hashes are additionally bound in each request and smoke/formal pairing.

## Frozen matrix

| Variable | Value / provenance |
|---|---|
| Protocols | nominal, id, ood_mild, ood_strong; evaluator PROTOCOLS controls all initial draws |
| Settings | fixed_dynamics, mutation |
| Evaluation seeds | [1409], current Table 2 canonical renderer |
| Episodes | 1 per walker / protocol / setting / eval seed |
| Horizon | 1000 environment decisions; existing TimeLimit |
| Mutation | step 250; native motor strength, ground friction, body mass + inertia |
| Post-mutation ranges | motor [0.6,1.4], friction [0.4,1.6], mass [0.6,1.4], drawn by native machinery |
| Recovery | x_vel; baseline window 25; fraction 0.8; min baseline 0.05; sustain 25 |
| Restoration / asymmetry / dual phase | frozen reference YAML values, unchanged; restoration 0, asymmetry false, dual phase false |
| Formal episode inventory | 97*4*2*1 = 776; smoke = 1*1*1*1 |

The adapter does not sample mutation or reset/step an evaluation environment. Standalone preflight creates native environments and observes nominal raw state without stepping; evaluation reseeds anew, so preflight RNG draws do not enter paired episodes.

## Metric contract

Let B be the native pre-mutation baseline, P the finite post samples starting at zero-based index 249, K=25. Native null/NaN handling is retained; serialized NaN/Inf or non-finite policy actions fail. Null recovery times are valid scientific missing outcomes, not failed numerical inference.

| Metric | Implementation / aggregation / window / censoring |
|---|---|
| Return | RecordEpisodeStatistics sum; evaluator summarize over episodes, then walkers; terminated episode retained |
| Length / survival | native episode l; recovery/adaptation survival uses timeout or length>=1000; early termination retained |
| Fall rate | **NOT_AVAILABLE** as a separately classified metric; native early-termination proxy must not be renamed fall |
| Velocity tracking | **NOT_USED** in this Table 2 profile (commands disabled); enabling a command task would change protocol |
| Sustained recovery rate | compute_sustained_recovery_event + summarize_recovery; denominator episodes reaching mutation, including censored / invalid-threshold outcomes |
| Recovery time | first inclusive K-step confirmation in post samples; first affected sample=1; recovered times only, null excluded and null counts recorded |
| Maximum normalized drop | compute_adaptation_metrics; max(0,B-min(P))/abs(B); B>=0.05 and finite samples required, otherwise null |
| Normalized post performance | mean(P)/abs(B); same validity; existing field post_perturbation_normalized_performance |
| Steady-state retention | mean(last min(K,len(P)) finite samples)/abs(B); same validity |
| NAR | mean(max(0,B-P))/abs(B); same validity |
| RMRT | restricted_recovery_time: observed time or 1000-250+1=751 for unrecovered; summarized native event values |
| Relapse | K consecutive outside recovery region after first confirmation; native denominator retained, does not revoke sustained recovery |

`_summarize_event_metric` excludes null/non-finite values and reports native counts. `summarize_recovery`, `summarize_adaptation`, and fixed-setting `build_cross_protocol_metrics` are called directly. Do not average success-only time as though it included all episodes. No new metric thresholds, fallback values, fall classifier, significance claim or new bootstrap is introduced.

## Output and resume

Existing raw schema v3 is assembled as `raw/fixed_dynamics.json` and `raw/mutation.json`, with native four-protocol `aggregate`, `per_walker`, and seed event records. Smoke has only ood_strong/mutation. Native per-episode JSONL traces retain rewards, done, x_vel/recovery signal, actions, mutation state/values, response boundary and native event fields. Policy-specific temporal latent is null and explicitly NOT_USED_STATIC_MODUMORPH; method stays ModuMorph, not MetaMorph-DR.

Additional request/config/manifest/provenance/preflight/status/log/binding and SHA256 COMPLETE records are sidecars. Existing downstream cell readers can consume consolidated JSON without changing metric code. This does not establish that every historical hardcoded 98-walker auditor accepts 97 IDs; use the current Strict-OOD97 consumers.

Resume requires identical request/source/asset hashes and explicit --resume. Completed protocol-setting cells and final evidence are hash-verified and skipped. Interrupted protocol cells restart in a new numbered attempt directory, preserving prior partial traces and logs; completed cells are never silently overwritten. All temporary outputs use project ./tmp. Formal rollout requires checksum-verified real smoke for the identical backend, checkpoint, config, identity and roots. No formal rollout was launched by Codex.
