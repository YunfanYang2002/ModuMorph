"""Reuse MorphAdapt evaluate_protocol unchanged, with a static policy backend."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from modumorph_policy_adapter import ModuMorphPolicyAdapter, PROPRIO_TYPES


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha(path):
    """Git text checkout CRLF/LF equivalence; assets/evidence use raw sha."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def git_identity(root):
    return {"sha": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip())}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def verify_complete(cell):
    recorded = json.loads((cell / "COMPLETE.json").read_text(encoding="utf-8"))
    if not recorded or any(sha(cell / name) != digest for name, digest in recorded.items()):
        raise ValueError(f"completed evidence changed: {cell}")


def preflight(args):
    import yaml
    contract_path = ROOT / "configs/modumorph_frozen_eval_binding.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    errors = []
    for name, digest in contract["source_sha256"].items():
        path = args.morphadapt_root / name
        if not path.is_file() or source_sha(path) != digest:
            errors.append(f"frozen evaluator source missing/changed: {name}")
    for name in ("checkpoint", "config", "strict_identity"):
        if not getattr(args, name).is_file():
            errors.append(f"missing {name}: {getattr(args, name)}")
    expected = (ROOT / "configs/modumorph_strict_ood97.txt").read_text(encoding="utf-8").splitlines()
    if source_sha(ROOT / "configs/modumorph_strict_ood97.txt") != contract["formal_evaluation"]["ood_walkers_sha256"]:
        errors.append("local Strict-OOD97 walker binding hash changed")
    rows = []
    if args.strict_identity.is_file():
        with args.strict_identity.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        required = {"walker_id", "family", "walker_xml", "walker_xml_sha256", "xml_cluster_id", "xml_cluster_size"}
        if not rows or not required.issubset(rows[0]):
            errors.append("Strict-OOD97 identity schema mismatch")
            rows = []
        elif len(rows) != 97 or {r["walker_id"] for r in rows} != set(expected) or len({r["xml_cluster_id"] for r in rows}) != 87:
            errors.append("identity is not the frozen 97 IDs / 87 XML clusters")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config.is_file() else None
    training_xmls = sorted(args.train_xml_root.glob("*.xml"))
    if not training_xmls:
        errors.append(f"missing/empty authoritative train XML root: {args.train_xml_root}")
    train_hashes = {sha(p) for p in training_xmls}
    if config is not None:
        train_ids = config["ENV"]["WALKERS"]
        if not train_ids or set(train_ids) != {p.stem for p in training_xmls}:
            errors.append("ModuMorph resolved config training IDs must match supplied train XML inventory")
    overlap = []
    actual_clusters = {}
    for row in rows:
        path = args.walker_root / "xml" / (row["walker_id"] + ".xml")
        metadata = args.walker_root / "metadata" / (row["walker_id"] + ".json")
        if not path.is_file():
            errors.append(f"missing evaluation XML: {path}")
            continue
        if not metadata.is_file():
            errors.append(f"missing morphology metadata: {metadata}")
        digest = sha(path)
        if digest != row["walker_xml_sha256"]:
            errors.append(f"XML identity hash mismatch: {path}")
        if digest in train_hashes or row["walker_id"] in {p.stem for p in training_xmls}:
            overlap.append(row["walker_id"])
        row["walker_xml"] = str(path.resolve())
        row["metadata"] = str(metadata.resolve())
        row["metadata_sha256"] = sha(metadata) if metadata.is_file() else "UNKNOWN"
        actual_clusters.setdefault(digest, []).append(row)
    if rows and len(actual_clusters) != 87:
        errors.append("actual XML hash clusters are not 87")
    for cluster in actual_clusters.values():
        if len({r["xml_cluster_id"] for r in cluster}) != 1 or any(int(r["xml_cluster_size"]) != len(cluster) for r in cluster):
            errors.append("identity cluster labels/sizes disagree with actual XML hashes")
    if overlap:
        errors.append(f"train/test morphology overlap: {overlap}")
    profile = args.morphadapt_root / contract["environment_config"]
    if not profile.is_file() or source_sha(profile) != contract["environment_config_sha256"]:
        errors.append("frozen environment config missing/changed")
    if args.walker is not None and args.walker not in expected:
        errors.append("smoke walker is outside Strict-OOD97")
    return contract, rows, errors, len(overlap)


def nominal_binding(env, adapter, walker):
    """Observe unfiltered node features and immutable nominal model only."""
    import numpy as np
    from tools.evaluate_dynamics import _single_unwrapped_env
    # Bypass only VecNormalize/VecPyTorch for this standalone binding observation.
    raw = env.venv.venv.envs[0].reset()
    physics = _single_unwrapped_env(env)
    model, agent = physics.sim.model, physics.modules["Agent"]
    nominal = physics._nominal_dynamics
    body, geom = agent.agent_body_idxs, agent.agent_geom_idxs
    if model.body_names[int(body[0])] != "torso/0":
        raise ValueError("first canonical agent body must be torso/0")
    snapshot = {key: getattr(model, key)[body].copy() for key in ("body_pos", "body_ipos", "body_iquat")}
    snapshot.update({"geom_quat": model.geom_quat[geom].copy(), "body_mass": nominal["body_mass"][body].copy(),
                     "geom_size": model.geom_size[geom].copy(), "geom_friction": nominal["geom_friction"][geom].copy(),
                     "jnt_pos": model.jnt_pos.copy(), "jnt_range": model.jnt_range.copy(), "jnt_axis": model.jnt_axis.copy(),
                     "actuator_gear": nominal["actuator_gear"].copy(), "dof_armature": model.dof_armature.copy(),
                     "dof_damping": model.dof_damping.copy()})
    mask, act_mask = raw["obs_padding_mask"], raw["act_padding_mask"]
    if int((~act_mask.astype(bool)).sum()) != model.nu:
        raise ValueError("actuator count differs from canonical valid slots")
    # Prove valid-slot extraction follows MuJoCo actuator/joint order.
    expected_joints = list(model.joint_names)[1:]
    actual_joints = [model.joint_names[int(i)] for i in model.actuator_trnid[:, 0]]
    if actual_joints != expected_joints:
        raise ValueError("MuJoCo actuators are not in the audited joint ordering")
    data = physics.sim.data
    data_snapshot = {key: getattr(data, key)[body].copy().tolist() for key in
                     ("body_xpos", "body_xquat", "body_xvelp", "body_xvelr")}
    data_snapshot.update(qpos=data.qpos.copy().tolist(), qvel=data.qvel.copy().tolist())
    context = adapter.bind(walker, raw["proprioceptive"], mask, act_mask,
                           {key: value.tolist() for key, value in snapshot.items()}, data_snapshot)
    return {"walker_id": walker, "context_sha256": hashlib.sha256(context.tobytes()).hexdigest(),
            "OBSERVATION_BINDING": "PASS", "ACTION_BINDING": "PASS", "PRIVILEGED_LEAKAGE": "NO",
            "actuator_joints": actual_joints, "valid_actuator_slots": np.flatnonzero(~act_mask.astype(bool)).tolist()}


def execute(args, contract, rows):
    import numpy as np
    # Select MorphAdapt namespace before any environment/model imports in host.
    sys.path.insert(0, str(args.morphadapt_root))
    import torch
    from metamorph.config import cfg, canonical_runtime_config_from_resolved
    from metamorph.algos.ppo.envs import make_vec_envs
    from metamorph.envs.vec_env.running_mean_std import RunningMeanStd
    from metamorph.utils import sample as su
    from tools import evaluate_dynamics as evaluator
    frozen = contract["formal_evaluation"]
    opts = ["ENV.WALKER_DIR", str(args.walker_root)]
    for key, value in contract["overrides"].items():
        opts.extend([key, str(value)])
    canonical_runtime_config_from_resolved(args.morphadapt_root / contract["environment_config"], opts)
    if cfg.MODEL.PROPRIOCEPTIVE_OBS_TYPES != PROPRIO_TYPES or cfg.ENV.VELOCITY_COMMAND.ENABLED or cfg.ENV.MOTION_COMMAND.ENABLED:
        raise ValueError("frozen observation profile differs from audited field semantics")
    cfg.DISTRIBUTED = False
    cfg.RANK = cfg.LOCAL_RANK = 0
    cfg.WORLD_SIZE = cfg.PPO.NUM_ENVS = 1
    cfg.PPO.CHECKPOINT_PATH = cfg.ADAPT.TEACHER_CHECKPOINT_PATH = ""
    cfg.VECENV.TYPE = "DummyVecEnv"
    cfg.OUT_DIR = str(args.output)
    cfg.DEVICE = args.device
    if cfg.ADAPT.ENABLED or cfg.ADAPT.EXPLICIT_ID.ENABLED or cfg.MIRROR_DATA_AUG:
        raise ValueError("static baseline environment must not expose adaptation/mirror inputs")
    attempt = len(list((args.output / "logs").glob("worker_*.log"))) + 1
    adapter = ModuMorphPolicyAdapter(ROOT, args.checkpoint, args.config, args.device, args.output / f"logs/worker_{attempt}.log")
    try:
        if adapter.max_limbs != cfg.MODEL.MAX_LIMBS:
            raise ValueError("checkpoint max_limbs differs from frozen evaluator")
        stats = adapter.metadata["rms"]
        rms = RunningMeanStd(shape=(adapter.max_limbs * 52,))
        rms.mean, rms.var, rms.count = np.asarray(stats["mean"]), np.asarray(stats["var"]), stats["count"]
        if not np.isfinite(rms.mean).all() or not np.isfinite(rms.var).all() or np.any(rms.var < 0) or rms.count <= 0:
            raise ValueError("invalid checkpoint normalization statistics")
        provenance = {"checkpoint_absolute_path": str(args.checkpoint), "checkpoint_filename": args.checkpoint.name,
                      "checkpoint_sha256": sha(args.checkpoint), "config_path": str(args.config), "config_sha256": sha(args.config),
                      "modumorph_git": git_identity(ROOT), "morphadapt_git": git_identity(args.morphadapt_root),
                      "model": {k: v for k, v in adapter.metadata.items() if k != "rms"}}
        import gym
        import mujoco_py
        provenance["runtime"] = {"hostname": platform.node(), "python": sys.version, "torch": torch.__version__,
            "numpy": np.__version__, "gym": gym.__version__, "mujoco_py": mujoco_py.__version__,
            "logical_device": args.device, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "UNKNOWN")}
        write_json(args.output / "provenance.json", provenance)
        walkers = [args.walker or rows[0]["walker_id"]] if args.mode == "smoke" else [r["walker_id"] for r in rows]
        bindings = []
        cfg.DYNAMICS.MID_EPISODE_ENABLED = False
        evaluator.configure_protocol("nominal")
        for walker in walkers:
            cfg.ENV.WALKERS = [walker]
            cfg.RNG_SEED = contract["evaluation_seeds"][0]
            su.set_seed(cfg.RNG_SEED)
            env = make_vec_envs(training=False, norm_rew=False, num_env=1)
            try:
                bindings.append(nominal_binding(env, adapter, walker))
            finally:
                env.close()
        write_json(args.output / "observation_bindings.json", bindings)
        # Backend-specific trace inference only. No invented Student latent.
        original_latent = evaluator.student_policy_latent
        evaluator.student_policy_latent = lambda model, obs, **kw: None if model is adapter else original_latent(model, obs, **kw)
        try:
            protocols = ["ood_strong"] if args.mode == "smoke" else frozen["protocols"]
            settings = ["mutation"] if args.mode == "smoke" else frozen["settings"]
            for setting in settings:
                cfg.DYNAMICS.MID_EPISODE_ENABLED = setting == "mutation"
                cfg.DYNAMICS.MID_EPISODE_STEP = frozen["mutation_step"]
                for protocol in protocols:
                    # Whole protocol cells are the resume unit; completed outputs never rewritten.
                    tag = f"{setting}_{protocol}"
                    completed = list((args.output / "raw").glob(tag + "_attempt_*/COMPLETE.json"))
                    if len(completed) > 1:
                        raise ValueError(f"multiple completed cells for {tag}")
                    if completed:
                        verify_complete(completed[0].parent)
                        print(f"SKIP COMPLETE {tag}", flush=True)
                        continue
                    attempts = len(list((args.output / "raw").glob(tag + "_attempt_*"))) + 1
                    cell = args.output / "raw" / f"{tag}_attempt_{attempts}"
                    complete = cell / "COMPLETE.json"
                    cell.mkdir(parents=True)
                    # Per-walker binding selection occurs before frozen inference, not stepping.
                    class BoundPolicy:
                        def __call__(self, obs):
                            adapter.reset(cfg.ENV.WALKERS[0])
                            return adapter(obs)
                    policy = BoundPolicy()
                    evaluator.student_policy_latent = lambda model, obs, **kw: None if model is policy else original_latent(model, obs, **kw)
                    identity = {"training_seed": "UNKNOWN", "method": "ModuMorph", "model_role": "static_morphology_policy",
                                "setting": setting, "checkpoint_sha256": sha(args.checkpoint), "config_sha256": sha(args.config),
                                "walkers_file_sha256": sha(ROOT / "configs/modumorph_strict_ood97.txt")}
                    eval_args = SimpleNamespace(episodes_per_walker=frozen["episodes_per_walker"],
                        max_steps_per_walker=frozen["max_steps_per_walker"], trace_out=str(cell / "traces"), trace_identity=identity,
                        replay_out=None, action_history_intervention="aligned", post_response_intervention=False)
                    result = evaluator.evaluate_protocol(policy, {"proprioceptive": rms}, protocol, walkers,
                                                         contract["evaluation_seeds"], eval_args)
                    raw = {"schema_version": 3, "config": str(args.config), "config_sha256": sha(args.config),
                           "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha(args.checkpoint),
                           "walkers_file": str(ROOT / "configs/modumorph_strict_ood97.txt"),
                           "walkers_file_sha256": identity["walkers_file_sha256"], "seeds": contract["evaluation_seeds"],
                           "episodes_per_walker_per_seed": frozen["episodes_per_walker"], "deterministic": True,
                           "mid_episode_perturbation": {"enabled": setting == "mutation", "step": frozen["mutation_step"],
                                "recovery_metric": cfg.DYNAMICS.RECOVERY_METRIC, "restore_step": cfg.DYNAMICS.MID_EPISODE_RESTORE_STEP,
                                "recovery_baseline_window": cfg.DYNAMICS.RECOVERY_BASELINE_WINDOW,
                                "recovery_threshold_fraction": cfg.DYNAMICS.RECOVERY_THRESHOLD_FRACTION,
                                "recovery_min_baseline": cfg.DYNAMICS.RECOVERY_MIN_BASELINE,
                                "recovery_sustain_steps": cfg.DYNAMICS.RECOVERY_SUSTAIN_STEPS,
                                "target_velocity_range": list(cfg.ENV.VELOCITY_COMMAND.RANGE),
                                "motor_strength": cfg.DYNAMICS.MID_EPISODE_MOTOR_STRENGTH_RANGE,
                                "motor_asymmetric": cfg.DYNAMICS.MID_EPISODE_MOTOR_ASYMMETRIC,
                                "motor_offset_range": list(cfg.DYNAMICS.MID_EPISODE_MOTOR_OFFSET_RANGE),
                                "motor_strength_bounds": list(cfg.DYNAMICS.MID_EPISODE_MOTOR_STRENGTH_BOUNDS),
                                "friction": cfg.DYNAMICS.MID_EPISODE_FRICTION_RANGE,
                                "mass": cfg.DYNAMICS.MID_EPISODE_MASS_RANGE},
                           "checkpoint_compatibility_fixes": [],
                           "adaptation_metrics": contract["adaptation_metrics"],
                           "sustained_recovery_metrics": contract["sustained_recovery_metrics"],
                           "protocols": {protocol: result}}
                    write_json(cell / "results.json", raw)
                    validate_smoke(cell, raw, frozen, args.mode)
                    # Trace schema stays native; correct backend-specific latent metadata.
                    for metadata in (cell / "traces").rglob("metadata.json"):
                        value = json.loads(metadata.read_text(encoding="utf-8"))
                        value.update(student_latent_status="NOT_USED_STATIC_MODUMORPH", student_latent_source="NOT_USED",
                                     student_latent_reason="Static morphology policy has no temporal Student latent")
                        write_json(metadata, value)
                    files = [p for p in cell.rglob("*") if p.is_file()]
                    write_json(complete, {str(p.relative_to(cell)): sha(p) for p in files})
                    print(f"COMPLETE {tag}", flush=True)
            for setting in settings:
                combined = None
                for protocol in protocols:
                    completed = list((args.output / "raw").glob(f"{setting}_{protocol}_attempt_*/COMPLETE.json"))
                    if len(completed) != 1:
                        raise ValueError("cannot assemble results without one completed cell per protocol")
                    value = json.loads((completed[0].parent / "results.json").read_text(encoding="utf-8"))
                    if combined is None:
                        combined = value
                    else:
                        combined["protocols"].update(value["protocols"])
                if setting == "fixed_dynamics":
                    combined["cross_protocol_metrics"] = evaluator.build_cross_protocol_metrics(combined["protocols"])
                write_json(args.output / "raw" / f"{setting}.json", combined)
        finally:
            evaluator.student_policy_latent = original_latent
    finally:
        adapter.close()


def validate_smoke(cell, raw, frozen, mode):
    for protocol in raw["protocols"].values():
        for walker in protocol["per_walker"].values():
            for record in walker["seeds"].values():
                if len(record["returns"]) != 1 or len(record["lengths"]) != 1:
                    raise ValueError("expected one episode per paired condition")
                if mode == "smoke" and not record.get("adaptation"):
                    raise ValueError("smoke did not reach mutation; cannot certify mutation execution")
    paths = list((cell / "traces").rglob("episodes.jsonl"))
    if not paths:
        raise ValueError("missing native per-step traces")
    for path in paths:
        episode = json.loads(path.read_text(encoding="utf-8").strip())
        steps = episode["steps"]
        if not steps or not (steps[-1]["terminated"] or steps[-1]["truncated"]):
            raise ValueError("trace does not end with a completed episode")
        if len(steps) != episode["episode_length"] or any(step["timestep"] != index for index, step in enumerate(steps)):
            raise ValueError("trace episode length or contiguous timestep schema mismatch")
        for step in steps:
            for key in ("step_reward", "measured_forward_velocity", "recovery_performance_metric", "policy_action"):
                # Native schema/serializer owns numeric validation. Require existing fields.
                if key not in step:
                    raise ValueError(f"native trace is missing {key}")
        if mode == "smoke":
            mutation = episode["mutation"]
            if mutation is None or mutation["step"] != frozen["mutation_step"]:
                raise ValueError("smoke mutation step mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--morphadapt-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="resolved training config.yaml")
    parser.add_argument("--strict-identity", type=Path, required=True)
    parser.add_argument("--walker-root", type=Path, required=True)
    parser.add_argument("--train-xml-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="project-relative ./tmp/... only")
    parser.add_argument("--mode", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--walker")
    parser.add_argument("--device", default="cuda:0", help="logical device; select physical GPU through CUDA_VISIBLE_DEVICES")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-evidence", type=Path, help="required completed real smoke for formal rollout")
    args = parser.parse_args()
    if args.output.is_absolute() or not args.output.parts or args.output.parts[0] != "tmp":
        parser.error("--output must be project-relative ./tmp/...; system /tmp is forbidden")
    args.output = (ROOT / args.output).resolve()
    if (ROOT / "tmp").resolve() not in args.output.parents:
        parser.error("--output escapes project ./tmp")
    (ROOT / "tmp").mkdir(exist_ok=True)
    for name in ("TMPDIR", "TMP", "TEMP"):
        os.environ[name] = str(ROOT / "tmp")
    for name in ("morphadapt_root", "checkpoint", "config", "strict_identity", "walker_root", "train_xml_root"):
        setattr(args, name, getattr(args, name).resolve())
    contract, rows, errors, overlap = preflight(args)
    request = {"checkpoint_sha256": sha(args.checkpoint) if args.checkpoint.is_file() else "UNKNOWN",
               "config_sha256": sha(args.config) if args.config.is_file() else "UNKNOWN",
               "identity_sha256": sha(args.strict_identity) if args.strict_identity.is_file() else "UNKNOWN",
               "contract_sha256": sha(ROOT / "configs/modumorph_frozen_eval_binding.json"),
               "walker_manifest_sha256": sha(ROOT / "configs/modumorph_strict_ood97.txt"),
               "train_xml_sha256": {p.name: sha(p) for p in sorted(args.train_xml_root.glob("*.xml"))},
               "metadata_sha256": {r["walker_id"]: r.get("metadata_sha256", "UNKNOWN") for r in rows},
               "modumorph_source_sha256": {name: source_sha(ROOT / name) for name in
                   ("tools/modumorph_policy_adapter.py", "tools/modumorph_inference_worker.py", "tools/run_modumorph_frozen_eval.py",
                    "metamorph/config.py", "metamorph/algos/ppo/model.py", "metamorph/algos/ppo/transformer.py",
                    "metamorph/envs/modules/agent.py", "metamorph/utils/model.py")},
               "walker_root": str(args.walker_root), "train_xml_root": str(args.train_xml_root), "mode": args.mode, "walker": args.walker}
    if args.output.exists():
        if not args.resume or not (args.output / "request.json").is_file():
            raise FileExistsError(f"refusing existing output: {args.output}")
        if json.loads((args.output / "request.json").read_text(encoding="utf-8")) != request:
            raise ValueError("resume request/provenance mismatch")
        if (args.output / "status.txt").read_text().strip() == "COMPLETE":
            verify_complete(args.output)
            print("SKIP COMPLETE RUN", flush=True)
            return 0
    else:
        args.output.mkdir(parents=True)
    (args.output / "logs").mkdir(exist_ok=True)
    write_json(args.output / "request.json", request)
    write_json(args.output / "manifest.json", {"schema_version": 1, "split": "Strict-OOD97", "morphologies": rows,
              "MORPHOLOGY_SPLIT_BINDING": "PASS" if not errors else "FAIL", "TRAIN_TEST_OVERLAP": overlap if rows and not errors else "UNKNOWN"})
    write_json(args.output / "config.json", contract)
    write_json(args.output / "preflight.json", {"errors": errors, "ASSET_PREFLIGHT": "FAIL" if errors else "PASS",
               "INFERENCE_BINDING": "SERVER_REQUIRED", "ROLLOUT_LAUNCHED": "NO"})
    if args.dry_run:
        (args.output / "status.txt").write_text("PREFLIGHT_FAILED\n" if errors else "PREFLIGHT_PASS\n")
        print(json.dumps({"errors": errors, "ROLLOUT_LAUNCHED": "NO", "output": str(args.output)}, indent=2))
        return 1 if errors else 0
    if errors:
        (args.output / "status.txt").write_text("FAILED\n")
        raise RuntimeError("preflight failed: " + "; ".join(errors))
    (args.output / "status.txt").write_text("RUNNING\n")
    try:
        if args.mode == "formal":
            if args.smoke_evidence is None:
                raise ValueError("formal rollout requires --smoke-evidence from real COMPLETE smoke")
            smoke = args.smoke_evidence.resolve()
            if (smoke / "status.txt").read_text().strip() != "COMPLETE":
                raise ValueError("server smoke has not completed")
            verify_complete(smoke)
            smoke_request = json.loads((smoke / "request.json").read_text())
            for key in ("checkpoint_sha256", "config_sha256", "identity_sha256", "contract_sha256", "walker_manifest_sha256",
                        "modumorph_source_sha256", "train_xml_sha256", "metadata_sha256", "walker_root", "train_xml_root"):
                if smoke_request[key] != request[key]:
                    raise ValueError(f"smoke evidence provenance mismatch: {key}")
            if smoke_request["mode"] != "smoke":
                raise ValueError("required evidence is not smoke")
        execute(args, contract, rows)
    except BaseException:
        (args.output / "status.txt").write_text("FAILED\n")
        failure = len(list((args.output / "logs").glob("failure_*.txt"))) + 1
        (args.output / f"logs/failure_{failure}.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    (args.output / "status.txt").write_text("COMPLETE\n")
    write_json(args.output / "COMPLETE.json", {str(p.relative_to(args.output)): sha(p) for p in args.output.rglob("*") if p.is_file() and p != args.output / "COMPLETE.json"})
    print(f"OUTPUT_DIR={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
