"""Local tests for the prepared frozen-evaluator behavioral regression."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_modumorph_evaluator_behavior_regression as regression


def episode(reward=1.0, action=None):
    action = [0.1, 0.2] if action is None else action
    return {"identity": {"training_seed": 1409, "eval_seed": 1409, "method": "state_action",
            "walker": "walker-a", "walker_id": "walker-a", "protocol": "ood_strong", "setting": "mutation",
            "checkpoint_sha256": "checkpoint", "config_sha256": "config"},
        "episode_return": reward, "episode_length": 1,
        "initial_dynamics": {"motor_strength": 1.0, "friction": 1.0, "mass": 1.0},
        "mutation": {"step": 250, "motor_strength": 0.6, "friction": 0.4, "mass": 1.4, "source": "terminal_episode_info"},
        "recovery": {"recovered": True, "recovery_time": 3, "retention": 0.5},
        "adaptation": {"pnp": 0.7, "nar": 0.2, "rmrt": 4.0}, "response_boundary": None,
        "steps": [{"timestep": 0, "mutation_step": 250, "post_mutation": False, "mutation_applied": False,
            "step_reward": reward, "measured_forward_velocity": 0.4, "recovery_performance_metric": 0.4,
            "student_policy_latent": [0.3], "policy_action": action, "action_history_used": [[0.0, 0.1]],
            "teacher_policy_latent": None, "teacher_student_l2_error": None,
            "privileged_context_before_action": None, "privileged_context_after_step": None,
            "actual_dynamics_before_action": None, "actual_dynamics_after_step": None,
            "terminated": True, "truncated": False, "termination_reason": "environment_terminal_reason_unavailable"}]}


def pair(payload):
    walker = {"return": {"mean": payload["episode_return"]}, "length": {"mean": payload["episode_length"]},
              "seeds": {"1409": {"returns": [payload["episode_return"]], "lengths": [payload["episode_length"]]}},
              "recovery": payload["recovery"], "adaptation": payload["adaptation"]}
    return {"episode": payload, "walker_record": walker}


def write_reference_fixture(root, contract):
    for folder in ("manifests", "raw", "commands"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    checkpoint, config = root / "checkpoint.pt", root / "config.yaml"
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("config\n", encoding="utf-8")
    run = {"seed": "1409", "method": "state_action", "checkpoint": str(checkpoint),
           "checkpoint_sha256": regression.sha(checkpoint), "config": str(config),
           "config_sha256": regression.sha(config), "training_manifest": "unused"}
    with (root / "manifests/canonical_runs.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=run, delimiter="\t")
        writer.writeheader()
        writer.writerow(run)
    walkers = (ROOT / "configs/modumorph_strict_ood97.txt").read_text(encoding="utf-8").splitlines()
    fields = ("walker_id", "family", "walker_xml", "walker_xml_sha256", "xml_cluster_id", "xml_cluster_size")
    with (root / "manifests/strict_ood97_identity.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index, walker in enumerate(walkers):
            writer.writerow({"walker_id": walker, "family": "floor", "walker_xml": "unused",
                "walker_xml_sha256": "xml", "xml_cluster_id": f"cluster-{min(index, 86)}", "xml_cluster_size": "1"})
    walker = walkers[0]
    walker_record = {"return": {"mean": 1.0}, "length": {"mean": 300.0},
        "seeds": {"1409": {"returns": [1.0], "lengths": [300],
            "dynamics": [{"motor_strength": 1.0, "friction": 1.0, "mass": 1.0}],
            "recovery": [{"parameters": {"motor_strength": 0.6, "friction": 0.4, "mass": 1.4}}],
            "adaptation": [{"pnp": 0.7, "nar": 0.2}]}},
        "recovery": {"count": 1}, "adaptation": {"count": 1}}
    result_path = root / "raw/reference.json"
    result = {"checkpoint_sha256": run["checkpoint_sha256"], "config_sha256": run["config_sha256"],
        "seeds": [1409], "mid_episode_perturbation": {"enabled": True, "step": 250},
        "protocols": {"ood_strong": {"per_walker": {walker: walker_record}}}}
    result_path.write_text(json.dumps(result), encoding="utf-8")
    command = ["python", "-u", "tools/evaluate_dynamics.py", "--cfg", str(config), "--checkpoint", str(checkpoint),
        "--walkers-file", "walkers.txt", "--protocols", "nominal,id,ood_mild,ood_strong", "--seeds", "1409",
        "--episodes-per-walker", "1", "--max-steps-per-walker", "1000", "--out", str(result_path),
        "--mid-episode-perturbation", "--mid-episode-step", "250", "ENV.WALKER_DIR", str(root / "walker-root"),
        "DYNAMICS.RECOVERY_BASELINE_WINDOW", "25", "DYNAMICS.RECOVERY_THRESHOLD_FRACTION", "0.8",
        "DYNAMICS.RECOVERY_MIN_BASELINE", "0.05", "DYNAMICS.RECOVERY_SUSTAIN_STEPS", "25",
        "DYNAMICS.MID_EPISODE_MOTOR_STRENGTH_RANGE", "[0.6, 1.4]",
        "DYNAMICS.MID_EPISODE_FRICTION_RANGE", "[0.4, 1.6]", "DYNAMICS.MID_EPISODE_MASS_RANGE", "[0.6, 1.4]"]
    rendered = [{"seed": 1409, "method": "state_action", "setting": "mutation", "raw": str(result_path), "command": command}]
    (root / "commands/rendered_commands.json").write_text(json.dumps(rendered), encoding="utf-8")
    (root / "commands/reference.sh").write_text("python tools/evaluate_dynamics.py\n", encoding="utf-8")
    return run, walker, walker_record, result_path


class ProvenanceContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = regression.load_json(regression.CONTRACT_PATH)
        self.binding = regression.load_json(regression.BINDING_PATH)

    def test_transient_hashes_are_preserved_but_not_authority(self):
        self.assertFalse(self.contract["table2_authority_for_transient_hashes"])
        self.assertEqual(self.contract["transient_hash_recovery"], "not_recoverable")
        for name, digest in self.contract["superseded_transient_hashes"].items():
            self.assertEqual(self.binding["superseded_transient_hashes"][name], digest)
            self.assertEqual(self.binding["source_sha256"][name], self.contract["candidate_source_sha256"][name])
            self.assertNotEqual(self.binding["source_sha256"][name], digest)
        self.assertEqual(self.binding["source_authority"], "reproducible_git_source_plus_behavior_regression")
        self.assertEqual(self.contract["candidate_authority_status"], "approved_behavior_regression_pass")
        self.assertEqual(self.binding["behavior_regression_evidence"],
                         self.contract["approved_behavior_regression_evidence"] |
                         {"reproducible_evaluator_source": self.contract["candidate_sha"],
                          "frozen_evaluator_behavior_regression": "PASS"})

    def test_3893388_git_source_gate_passes_and_current_drift_fails(self):
        counterpart = ROOT.parent / "rmamorph"
        expected = regression.source_expectations(self.contract, self.binding)
        self.assertEqual(regression.verify_git_source(counterpart, self.contract["candidate_sha"], expected), {})
        current = regression.verify_source_tree(counterpart, expected)
        self.assertEqual(set(current), set(self.contract["candidate_source_sha256"]))

    def test_continuous_trace_and_metrics_accept_strict_roundoff(self):
        reference = pair(episode())
        candidate = copy.deepcopy(reference)
        candidate["walker_record"]["return"]["mean"] += 1e-12
        candidate["walker_record"]["adaptation"]["pnp"] += 1e-12
        statistics = regression.compare_raw(reference["walker_record"], candidate["walker_record"], self.contract)
        self.assertIn("raw.per_walker", statistics)

    def test_continuous_1e8_mismatch_is_rejected_with_path(self):
        reference = pair(episode())
        candidate = copy.deepcopy(reference)
        candidate["walker_record"]["return"]["mean"] += 1e-8
        with self.assertRaisesRegex(AssertionError, r"raw.per_walker.*mean.*abs=.*rel="):
            regression.compare_raw(reference["walker_record"], candidate["walker_record"], self.contract)

    def test_raw_discrete_mismatch_is_rejected(self):
        reference = pair(episode())["walker_record"]
        for path in ("length", "count", "null"):
            candidate = copy.deepcopy(reference)
            if path == "length":
                candidate["seeds"]["1409"]["lengths"][0] += 1
            elif path == "count":
                candidate["recovery"]["recovered"] = False
            else:
                candidate["adaptation"]["missing"] = None
                reference["adaptation"]["missing"] = "missing"
            with self.subTest(path=path), self.assertRaises(AssertionError):
                regression.compare_raw(reference, candidate, self.contract)

    def test_level_a_fields_are_exact(self):
        mutations = (("episode_length", 2), ("mutation", {**episode()["mutation"], "motor_strength": 0.600000000001}))
        for key, value in mutations:
            reference = pair(episode())
            candidate = copy.deepcopy(reference)
            candidate["episode"][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                regression.compare_trace(reference, candidate, self.contract)
        for key, value in (("terminated", False), ("truncated", True), ("timestep", 1)):
            reference = pair(episode())
            candidate = copy.deepcopy(reference)
            candidate["episode"]["steps"][0][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                regression.compare_trace(reference, candidate, self.contract)
        reference = pair(episode())
        candidate = copy.deepcopy(reference)
        candidate["episode"]["identity"]["method"] = "state_only"
        with self.assertRaisesRegex(AssertionError, "State.Action MorphAdapt"):
            regression.compare_trace(reference, candidate, self.contract)

    def test_null_structure_and_nonfinite_are_rejected(self):
        reference = pair(episode())
        for value in (None, float("nan"), float("inf")):
            candidate = copy.deepcopy(reference)
            candidate["episode"]["steps"][0]["step_reward"] = value
            with self.subTest(value=value), self.assertRaises(AssertionError):
                    regression.compare_trace(reference, candidate, self.contract)

    def test_candidate_trace_proves_mutation_step_250(self):
        payload = episode()
        payload["episode_length"] = 250
        payload["steps"] = [copy.deepcopy(payload["steps"][0]) for _ in range(250)]
        for index, step in enumerate(payload["steps"]):
            step["timestep"] = index
            step["terminated"] = index == 249
        payload["steps"][249]["mutation_applied"] = True
        payload["steps"][249]["mutation_parameters"] = copy.deepcopy(payload["mutation"])
        reference = {"walker": "walker-a", "run": {"checkpoint_sha256": "checkpoint", "config_sha256": "config"}}
        regression.validate_candidate_trace(payload, reference, self.contract)
        for change in ("step", "event"):
            invalid = copy.deepcopy(payload)
            if change == "step":
                invalid["mutation"]["step"] = 251
            else:
                invalid["steps"][249]["mutation_applied"] = False
            with self.subTest(change=change), self.assertRaises(AssertionError):
                regression.validate_candidate_trace(invalid, reference, self.contract)

    def test_source_tree_gate_rejects_changed_content(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            root = Path(directory)
            target = root / "tools/evaluate_dynamics.py"
            target.parent.mkdir()
            target.write_text("candidate\n", encoding="utf-8")
            expected = {"tools/evaluate_dynamics.py": regression.sha(target, text=True)}
            self.assertEqual(regression.verify_source_tree(root, expected), {})
            target.write_text("arbitrary head\n", encoding="utf-8")
            self.assertEqual(set(regression.verify_source_tree(root, expected)), {"tools/evaluate_dynamics.py"})

    def test_raw_reference_selection_does_not_require_trace(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            root = Path(directory)
            run, walker, walker_record, result_path = write_reference_fixture(root, self.contract)
            selected = regression.select_reference(root, self.contract)
            self.assertEqual(selected["result_path"], result_path)
            self.assertEqual(selected["walker"], walker)
            self.assertEqual(selected["walker_record"], walker_record)
            self.assertEqual(selected["selection_basis"], "raw seed recovery/adaptation event")
            self.assertTrue(selected["mutation_parameters_available"])
            self.assertIsNone(regression.find_optional_trace(None, selected, self.contract))

    def test_raw_reference_rejects_wrong_authority_fields(self):
        for field in ("checkpoint", "config", "protocol", "setting"):
            with self.subTest(field=field), tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
                root = Path(directory)
                run, walker, _, result_path = write_reference_fixture(root, self.contract)
                result = json.loads(result_path.read_text())
                if field == "checkpoint":
                    result["checkpoint_sha256"] = "wrong"
                elif field == "config":
                    result["config_sha256"] = "wrong"
                elif field == "protocol":
                    result["protocols"] = {"id": result["protocols"]["ood_strong"]}
                else:
                    result["mid_episode_perturbation"]["enabled"] = False
                result_path.write_text(json.dumps(result), encoding="utf-8")
                with self.assertRaises(ValueError):
                    regression.select_reference(root, self.contract)

    def test_raw_reference_rejects_walker_outside_strict_ood97(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            root = Path(directory)
            _, walker, _, result_path = write_reference_fixture(root, self.contract)
            result = json.loads(result_path.read_text())
            record = result["protocols"]["ood_strong"]["per_walker"].pop(walker)
            result["protocols"]["ood_strong"]["per_walker"]["outside-walker"] = record
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no Strict-OOD97 walker"):
                regression.select_reference(root, self.contract)


if __name__ == "__main__":
    unittest.main(verbosity=2)
