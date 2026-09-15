"""Compare one existing frozen Table 2 episode with a reproducible evaluator replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/modumorph_evaluator_behavior_regression.json"
BINDING_PATH = ROOT / "configs/modumorph_frozen_eval_binding.json"
CONTINUOUS_STEP_FIELDS = (
    "step_reward", "measured_forward_velocity", "recovery_performance_metric",
    "policy_action", "student_policy_latent", "action_history_used",
    "target_velocity", "teacher_student_l2_error",
    "privileged_context_before_action", "privileged_context_after_step",
    "actual_dynamics_before_action", "actual_dynamics_after_step",
)


def sha(path, text=False):
    data = Path(path).read_bytes()
    if text:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def canonical_method(value):
    return "".join(character for character in str(value).lower() if character.isalnum())


def source_expectations(contract, binding):
    expected = dict(binding["source_sha256"])
    expected.update(contract["candidate_source_sha256"])
    return expected


def verify_source_tree(root, expected):
    mismatches = {}
    for name, digest in expected.items():
        path = Path(root) / name
        actual = sha(path, text=True) if path.is_file() else "MISSING"
        if actual != digest:
            mismatches[name] = {"expected": digest, "actual": actual}
    return mismatches


def verify_git_source(repo, commit, expected):
    mismatches = {}
    for name, digest in expected.items():
        try:
            data = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{name}"])
            actual = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
        except subprocess.CalledProcessError:
            actual = "MISSING"
        if actual != digest:
            mismatches[name] = {"expected": digest, "actual": actual}
    return mismatches


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def method_matches(value):
    return canonical_method(value) in {"stateaction", "morphadaptstateaction", "stateactionmorphadapt"}


def select_reference(reference_root, trace_root, contract):
    manifest = Path(reference_root) / "manifests/canonical_runs.tsv"
    if not manifest.is_file():
        raise FileNotFoundError("missing authoritative canonical_runs.tsv: " + str(manifest))
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    runs = [row for row in rows if int(row["seed"]) == contract["training_seed"] and method_matches(row["method"])]
    if len(runs) != 1:
        raise ValueError(f"expected one seed-1409 State+Action canonical run, found {len(runs)}")
    run = runs[0]
    checkpoint, config = Path(run["checkpoint"]), Path(run["config"])
    for label, path, expected in (("checkpoint", checkpoint, run["checkpoint_sha256"]),
                                  ("config", config, run["config_sha256"])):
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"reference {label} missing or SHA256 mismatch: {path}")

    candidates = []
    for path in sorted(Path(trace_root).rglob("episodes.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            episode = json.loads(line)
            identity, mutation = episode.get("identity", {}), episode.get("mutation")
            if (int(identity.get("training_seed", -1)) == contract["training_seed"]
                    and int(identity.get("eval_seed", -1)) == contract["evaluation_seed"]
                    and method_matches(identity.get("method", ""))
                    and identity.get("protocol") == contract["protocol"]
                    and identity.get("setting") == contract["setting"]
                    and mutation is not None and mutation.get("step") == contract["mutation_step"]
                    and episode.get("episode_length", 0) >= contract["mutation_step"]
                    and len(episode.get("steps", [])) == episode.get("episode_length")):
                candidates.append((str(path), line_number, episode))
    if not candidates:
        raise ValueError("no existing seed-1409 State+Action ood_strong mutation trace reached step 250")
    trace_path, line_number, episode = candidates[0]
    walker = episode["identity"].get("walker_id", episode["identity"].get("walker"))
    if not walker:
        raise ValueError("reference trace lacks walker identity")
    if episode["identity"].get("checkpoint_sha256") != run["checkpoint_sha256"]:
        raise ValueError("reference trace checkpoint does not match canonical run")
    if episode["identity"].get("config_sha256") != run["config_sha256"]:
        raise ValueError("reference trace config does not match canonical run")

    identity_path = Path(reference_root) / "manifests/strict_ood97_identity.tsv"
    with identity_path.open(encoding="utf-8", newline="") as handle:
        identities = list(csv.DictReader(handle, delimiter="\t"))
    identity_rows = [row for row in identities if row["walker_id"] == walker]
    if len(identity_rows) != 1:
        raise ValueError("selected walker is not unique in frozen Strict-OOD97 identity")
    identity = identity_rows[0]

    result_matches = []
    for path in sorted((Path(reference_root) / "raw").glob("*.json")):
        try:
            value = load_json(path)
            record = value["protocols"][contract["protocol"]]["per_walker"][walker]
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        if value.get("checkpoint_sha256") == run["checkpoint_sha256"] and value.get("config_sha256") == run["config_sha256"]:
            result_matches.append((path, value, record))
    if len(result_matches) != 1:
        raise ValueError(f"expected one authoritative raw result for selected episode, found {len(result_matches)}")
    result_path, result, walker_record = result_matches[0]
    return {"run": run, "checkpoint": checkpoint, "config": config, "walker": walker,
            "identity_path": identity_path, "identity": identity, "trace_path": Path(trace_path),
            "trace_line": line_number, "episode": episode, "result_path": result_path,
            "result": result, "walker_record": walker_record, "candidate_count": len(candidates)}


def exact(actual, expected, path):
    if type(actual) is not type(expected):
        raise AssertionError(f"{path}: type {type(actual).__name__} != {type(expected).__name__}")
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"{path}: keys {list(actual)} != {list(expected)}")
        for key in actual:
            exact(actual[key], expected[key], f"{path}[{key!r}]")
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: length {len(actual)} != {len(expected)}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            exact(left, right, f"{path}[{index}]")
    elif actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def compare_numeric(actual, expected, path, rtol, atol, statistics, group=None):
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: length {len(actual)} != {len(expected)}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_numeric(left, right, f"{path}[{index}]", rtol, atol, statistics, group)
    elif isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"{path}: keys {list(actual)} != {list(expected)}")
        for key in actual:
            compare_numeric(actual[key], expected[key], f"{path}[{key!r}]", rtol, atol, statistics, group)
    elif isinstance(actual, float) and isinstance(expected, float):
        if not math.isfinite(actual) or not math.isfinite(expected):
            raise AssertionError(f"{path}: non-finite value")
        absolute = abs(actual - expected)
        relative = absolute / abs(expected) if expected else (0.0 if absolute == 0 else math.inf)
        item = statistics.setdefault(group or path, {"max_abs_diff": 0.0, "max_rel_diff": 0.0})
        item["max_abs_diff"], item["max_rel_diff"] = max(item["max_abs_diff"], absolute), max(item["max_rel_diff"], relative)
        if absolute > atol + rtol * abs(expected):
            raise AssertionError(f"{path}: actual={actual!r} expected={expected!r} abs={absolute!r} rel={relative!r}")
    else:
        exact(actual, expected, path)


def compare(reference, candidate, contract):
    rtol, atol = contract["continuous_rtol"], contract["continuous_atol"]
    ref_episode, candidate_episode = reference["episode"], candidate["episode"]
    if not method_matches(ref_episode["identity"].get("method", "")) or not method_matches(candidate_episode["identity"].get("method", "")):
        raise AssertionError("episode.identity['method']: both traces must identify State+Action MorphAdapt")
    identity_keys = ("training_seed", "eval_seed", "walker", "walker_id", "protocol", "setting", "checkpoint_sha256", "config_sha256")
    for key in identity_keys:
        if key in ref_episode["identity"] or key in candidate_episode["identity"]:
            exact(candidate_episode["identity"].get(key), ref_episode["identity"].get(key), f"episode.identity[{key!r}]")
    for key in ("episode_length", "mutation"):
        exact(candidate_episode[key], ref_episode[key], f"episode[{key!r}]")
    if len(candidate_episode["steps"]) != len(ref_episode["steps"]):
        raise AssertionError("episode.steps: trace step count mismatch")
    statistics = {}
    for index, (actual, expected) in enumerate(zip(candidate_episode["steps"], ref_episode["steps"])):
        if actual.keys() != expected.keys():
            raise AssertionError(f"episode.steps[{index}]: keys differ")
        for key in actual:
            path = f"episode.steps[{index}][{key!r}]"
            if key in CONTINUOUS_STEP_FIELDS:
                compare_numeric(actual[key], expected[key], path, rtol, atol, statistics, "trace." + key)
            elif key == "mutation_parameters":
                exact(actual[key], expected[key], path)
            else:
                exact(actual[key], expected[key], path)
    for key in ("episode_return", "initial_dynamics", "recovery", "adaptation", "response_boundary"):
        compare_numeric(candidate_episode.get(key), ref_episode.get(key), f"episode[{key!r}]", rtol, atol, statistics, "episode." + key)
    compare_numeric(candidate["walker_record"], reference["walker_record"], "formal.per_walker", rtol, atol, statistics, "formal.per_walker")
    return statistics


def json_statistics(statistics):
    return {key: {name: (value if math.isfinite(value) else "Infinity") for name, value in item.items()}
            for key, item in statistics.items()}


def rendered_walker_root(reference_root, contract):
    path = Path(reference_root) / "commands/rendered_commands.json"
    commands = load_json(path)
    matches = [row for row in commands if int(row["seed"]) == contract["training_seed"]
               and method_matches(row["method"]) and row["setting"] == contract["setting"]]
    if len(matches) != 1:
        raise ValueError("authoritative rendered command is missing or ambiguous")
    command = matches[0]["command"]
    required = {"--protocols": "nominal,id,ood_mild,ood_strong", "--seeds": "1409",
                "--episodes-per-walker": "1", "--max-steps-per-walker": "1000",
                "--mid-episode-step": "250", "DYNAMICS.RECOVERY_BASELINE_WINDOW": "25",
                "DYNAMICS.RECOVERY_THRESHOLD_FRACTION": "0.8", "DYNAMICS.RECOVERY_MIN_BASELINE": "0.05",
                "DYNAMICS.RECOVERY_SUSTAIN_STEPS": "25", "DYNAMICS.MID_EPISODE_MOTOR_STRENGTH_RANGE": "[0.6, 1.4]",
                "DYNAMICS.MID_EPISODE_FRICTION_RANGE": "[0.4, 1.6]", "DYNAMICS.MID_EPISODE_MASS_RANGE": "[0.6, 1.4]"}
    if "--mid-episode-perturbation" not in command:
        raise ValueError("authoritative rendered command lacks mutation mode")
    for key, value in required.items():
        if key not in command or command[command.index(key) + 1] != value:
            raise ValueError(f"authoritative rendered command contract mismatch: {key}")
    key = "ENV.WALKER_DIR"
    if key not in command:
        raise ValueError("authoritative rendered command lacks walker root")
    return Path(command[command.index(key) + 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmamorph-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, help="authoritative existing trace root; defaults to reference root")
    parser.add_argument("--output", type=Path, default=Path("tmp/modumorph_evaluator_behavior_regression"))
    parser.add_argument("--worktree", type=Path, default=Path("tmp/rmamorph_behavior_regression_3893388"))
    args = parser.parse_args()
    for name in ("output", "worktree"):
        path = getattr(args, name)
        if path.is_absolute() or not path.parts or path.parts[0] != "tmp":
            parser.error(f"--{name} must be project-relative tmp/...; system /tmp is forbidden")
        setattr(args, name, (ROOT / path).resolve())
    args.rmamorph_root, args.reference_root = args.rmamorph_root.resolve(), args.reference_root.resolve()
    args.trace_root = args.trace_root.resolve() if args.trace_root else args.reference_root
    contract, binding = load_json(CONTRACT_PATH), load_json(BINDING_PATH)
    expected_sources = source_expectations(contract, binding)
    if binding["source_reference_sha"] != contract["candidate_sha"]:
        raise ValueError("binding and behavior candidate commit disagree")
    if binding["source_sha256"] | contract["candidate_source_sha256"] != expected_sources:
        raise AssertionError("candidate source expectation construction failed")
    transient = contract["superseded_transient_hashes"]
    if contract["table2_authority_for_transient_hashes"] or any(binding["source_sha256"].get(k) != v for k, v in transient.items()):
        raise ValueError("transient provenance declaration disagrees with preserved binding history")
    git_mismatches = verify_git_source(args.rmamorph_root, contract["candidate_sha"], expected_sources)
    if git_mismatches:
        raise ValueError("candidate commit source gate failed: " + json.dumps(git_mismatches, sort_keys=True))
    reference = select_reference(args.reference_root, args.trace_root, contract)
    walker_root = rendered_walker_root(args.reference_root, contract)
    xml = walker_root / "xml" / (reference["walker"] + ".xml")
    if not xml.is_file() or sha(xml) != reference["identity"]["walker_xml_sha256"]:
        raise ValueError("selected reference walker XML missing or SHA256 mismatch: " + str(xml))
    if args.output.exists():
        raise FileExistsError("refusing existing behavior regression output: " + str(args.output))
    args.output.mkdir(parents=True)
    (args.output / "status.txt").write_text("RUNNING\n", encoding="utf-8")
    write_json(args.output / "reference.json", {
        "reference_result_path": str(reference["result_path"]), "reference_trace_path": str(reference["trace_path"]),
        "reference_trace_line": reference["trace_line"], "reference_checkpoint": str(reference["checkpoint"]),
        "reference_checkpoint_sha256": reference["run"]["checkpoint_sha256"], "reference_config": str(reference["config"]),
        "reference_config_sha256": reference["run"]["config_sha256"], "walker_id": reference["walker"],
        "walker_xml": str(xml), "walker_xml_sha256": sha(xml), "eval_seed": contract["evaluation_seed"],
        "protocol": contract["protocol"], "setting": contract["setting"], "eligible_reference_count": reference["candidate_count"]})
    if args.worktree.exists():
        if subprocess.check_output(["git", "-C", str(args.worktree), "rev-parse", "HEAD"], text=True).strip() != contract["candidate_sha"]:
            raise ValueError("existing candidate worktree is at the wrong commit")
        if subprocess.check_output(["git", "-C", str(args.worktree), "status", "--porcelain"], text=True).strip():
            raise ValueError("existing candidate worktree is dirty")
    else:
        subprocess.run(["git", "-C", str(args.rmamorph_root), "worktree", "add", "--detach", str(args.worktree), contract["candidate_sha"]], check=True)
    tree_mismatches = verify_source_tree(args.worktree, expected_sources)
    if tree_mismatches:
        raise ValueError("detached candidate source gate failed: " + json.dumps(tree_mismatches, sort_keys=True))
    walker_file = args.output / "walker.txt"
    walker_file.write_text(reference["walker"] + "\n", encoding="utf-8")
    candidate_dir = args.output / "candidate"
    candidate_dir.mkdir()
    command = [sys.executable, "-u", "tools/evaluate_dynamics.py", "--cfg", str(reference["config"]),
        "--checkpoint", str(reference["checkpoint"]), "--walkers-file", str(walker_file),
        "--protocols", contract["protocol"], "--seeds", str(contract["evaluation_seed"]),
        "--episodes-per-walker", "1", "--max-steps-per-walker", str(contract["max_steps_per_walker"]),
        "--out", str(candidate_dir / "results.json"), "--mid-episode-perturbation", "--mid-episode-step",
        str(contract["mutation_step"]), "--trace-out", str(candidate_dir / "traces"),
        "--trace-training-seed", str(contract["training_seed"]), "--trace-method", contract["method"],
        "ENV.WALKER_DIR", str(walker_root), "DYNAMICS.RECOVERY_BASELINE_WINDOW", "25",
        "DYNAMICS.RECOVERY_THRESHOLD_FRACTION", "0.8", "DYNAMICS.RECOVERY_MIN_BASELINE", "0.05",
        "DYNAMICS.RECOVERY_SUSTAIN_STEPS", "25", "DYNAMICS.MID_EPISODE_MOTOR_STRENGTH_RANGE", "[0.6, 1.4]",
        "DYNAMICS.MID_EPISODE_FRICTION_RANGE", "[0.4, 1.6]", "DYNAMICS.MID_EPISODE_MASS_RANGE", "[0.6, 1.4]"]
    write_json(args.output / "request.json", {"candidate_sha": contract["candidate_sha"], "command": command,
        "source_sha256": expected_sources, "continuous_rtol": contract["continuous_rtol"],
        "continuous_atol": contract["continuous_atol"], "ROLLOUT_SCOPE": "ONE_EXISTING_REFERENCE_EPISODE"})
    subprocess.run(command, cwd=args.worktree, check=True)
    candidate_result = load_json(candidate_dir / "results.json")
    candidate_trace_paths = list((candidate_dir / "traces").rglob("episodes.jsonl"))
    if len(candidate_trace_paths) != 1:
        raise ValueError(f"expected one candidate trace file, found {len(candidate_trace_paths)}")
    lines = [line for line in candidate_trace_paths[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected one candidate trace episode, found {len(lines)}")
    candidate_episode = json.loads(lines[0])
    candidate_walker = candidate_result["protocols"][contract["protocol"]]["per_walker"][reference["walker"]]
    candidate = {"episode": candidate_episode, "walker_record": candidate_walker}
    report = {"TRANSIENT_BINDING_HASH_BUG": "YES", "TABLE2_AUTHORITY_FOR_88FC_HASH": "NO",
        "TABLE2_AUTHORITY_FOR_76C70_HASH": "NO", "RECOVERABLE_TRANSIENT_BYTES": "NO",
        "BEHAVIOR_REGRESSION_REFERENCE": str(reference["trace_path"]),
        "REPRODUCIBLE_CANDIDATE_SHA": contract["candidate_sha"], "SOURCE_INTEGRITY_GATE_PRESERVED": "YES"}
    try:
        statistics = compare(reference, candidate, contract)
    except AssertionError as error:
        report.update(FROZEN_EVALUATOR_BEHAVIOR_REGRESSION="FAIL", BINDING_CONTRACT_REPAIRED="NO",
                      FIRST_BEHAVIORAL_MISMATCH=str(error), MODUMORPH_FORMAL_EVAL_BLOCKED="YES")
        write_json(args.output / "report.json", report)
        (args.output / "status.txt").write_text("FAILED\n", encoding="utf-8")
        print("FROZEN_EVALUATOR_BEHAVIOR_REGRESSION=FAIL")
        print("FIRST_BEHAVIORAL_MISMATCH=" + str(error))
        raise
    report.update(FROZEN_EVALUATOR_BEHAVIOR_REGRESSION="PASS", BINDING_CONTRACT_REPAIRED="NO_PENDING_LOCAL_COMMIT",
                  CONTINUOUS_DIFFERENCES=json_statistics(statistics), MODUMORPH_FORMAL_EVAL_BLOCKED="YES_PENDING_BINDING_REPAIR")
    report_path = args.output / "report.json"
    write_json(report_path, report)
    write_json(args.output / "binding_repair_proposal.json", {
        "source_authority": "reproducible_git_source_plus_behavior_regression",
        "source_reference_sha": contract["candidate_sha"], "replacement_source_sha256": contract["candidate_source_sha256"],
        "superseded_transient_hashes": transient, "transient_hash_recovery": "not_recoverable",
        "behavior_regression_evidence": {"path": str(report_path), "sha256": sha(report_path)}})
    (args.output / "status.txt").write_text("COMPLETE\n", encoding="utf-8")
    files = [path for path in args.output.rglob("*") if path.is_file() and path.name != "COMPLETE.json"]
    write_json(args.output / "COMPLETE.json", {str(path.relative_to(args.output)): sha(path) for path in files})
    print("FROZEN_EVALUATOR_BEHAVIOR_REGRESSION=PASS")
    print("REPRODUCIBLE_EVALUATOR_SOURCE=" + contract["candidate_sha"])
    print("OUTPUT_DIR=" + str(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
