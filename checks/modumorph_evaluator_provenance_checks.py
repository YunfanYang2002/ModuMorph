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


class ProvenanceContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = regression.load_json(regression.CONTRACT_PATH)
        self.binding = regression.load_json(regression.BINDING_PATH)

    def test_transient_hashes_are_preserved_but_not_authority(self):
        self.assertFalse(self.contract["table2_authority_for_transient_hashes"])
        self.assertEqual(self.contract["transient_hash_recovery"], "not_recoverable")
        for name, digest in self.contract["superseded_transient_hashes"].items():
            self.assertEqual(self.binding["source_sha256"][name], digest)
            self.assertNotEqual(self.contract["candidate_source_sha256"][name], digest)
        self.assertEqual(self.contract["candidate_authority_status"], "pending_behavior_regression")

    def test_3893388_git_source_gate_passes_and_current_drift_fails(self):
        counterpart = ROOT.parent / "rmamorph"
        expected = regression.source_expectations(self.contract, self.binding)
        self.assertEqual(regression.verify_git_source(counterpart, self.contract["candidate_sha"], expected), {})
        current = regression.verify_source_tree(counterpart, expected)
        self.assertEqual(set(current), set(self.contract["candidate_source_sha256"]))

    def test_continuous_trace_and_metrics_accept_strict_roundoff(self):
        reference = pair(episode())
        candidate = copy.deepcopy(reference)
        candidate["episode"]["steps"][0]["step_reward"] += 1e-12
        candidate["episode"]["steps"][0]["policy_action"][1] += 1e-12
        candidate["episode"]["adaptation"]["pnp"] += 1e-12
        candidate["walker_record"]["return"]["mean"] += 1e-12
        statistics = regression.compare(reference, candidate, self.contract)
        self.assertIn("trace.step_reward", statistics)
        self.assertIn("trace.policy_action", statistics)
        self.assertIn("formal.per_walker", statistics)

    def test_continuous_1e8_mismatch_is_rejected_with_path(self):
        reference = pair(episode())
        candidate = copy.deepcopy(reference)
        candidate["episode"]["steps"][0]["measured_forward_velocity"] += 1e-8
        with self.assertRaisesRegex(AssertionError, r"episode.steps\[0\].*measured_forward_velocity.*abs=.*rel="):
            regression.compare(reference, candidate, self.contract)

    def test_level_a_fields_are_exact(self):
        mutations = (("episode_length", 2), ("mutation", {**episode()["mutation"], "motor_strength": 0.600000000001}))
        for key, value in mutations:
            reference = pair(episode())
            candidate = copy.deepcopy(reference)
            candidate["episode"][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                regression.compare(reference, candidate, self.contract)
        for key, value in (("terminated", False), ("truncated", True), ("timestep", 1)):
            reference = pair(episode())
            candidate = copy.deepcopy(reference)
            candidate["episode"]["steps"][0][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                regression.compare(reference, candidate, self.contract)
        reference = pair(episode())
        candidate = copy.deepcopy(reference)
        candidate["episode"]["identity"]["method"] = "state_only"
        with self.assertRaisesRegex(AssertionError, "State.Action MorphAdapt"):
            regression.compare(reference, candidate, self.contract)

    def test_null_structure_and_nonfinite_are_rejected(self):
        reference = pair(episode())
        for value in (None, float("nan"), float("inf")):
            candidate = copy.deepcopy(reference)
            candidate["episode"]["steps"][0]["step_reward"] = value
            with self.subTest(value=value), self.assertRaises(AssertionError):
                regression.compare(reference, candidate, self.contract)

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

    def test_reference_selection_skips_episode_before_mutation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            root = Path(directory)
            for folder in ("manifests", "raw", "traces/early", "traces/eligible"):
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
            with (root / "manifests/strict_ood97_identity.tsv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("walker_id", "family", "walker_xml", "walker_xml_sha256",
                                                               "xml_cluster_id", "xml_cluster_size"), delimiter="\t")
                writer.writeheader()
                writer.writerow({"walker_id": "walker-a", "family": "floor", "walker_xml": "unused",
                    "walker_xml_sha256": "xml", "xml_cluster_id": "cluster", "xml_cluster_size": "1"})
            eligible = episode()
            eligible["identity"]["checkpoint_sha256"] = run["checkpoint_sha256"]
            eligible["identity"]["config_sha256"] = run["config_sha256"]
            eligible["episode_length"] = 250
            eligible["steps"] = [copy.deepcopy(eligible["steps"][0]) for _ in range(250)]
            for index, step in enumerate(eligible["steps"]):
                step["timestep"] = index
            early = copy.deepcopy(eligible)
            early["episode_length"], early["steps"], early["mutation"] = 100, early["steps"][:100], None
            (root / "traces/early/episodes.jsonl").write_text(json.dumps(early) + "\n", encoding="utf-8")
            target = root / "traces/eligible/episodes.jsonl"
            target.write_text(json.dumps(eligible) + "\n", encoding="utf-8")
            result = {"checkpoint_sha256": run["checkpoint_sha256"], "config_sha256": run["config_sha256"],
                      "protocols": {"ood_strong": {"per_walker": {"walker-a": pair(eligible)["walker_record"]}}}}
            (root / "raw/reference.json").write_text(json.dumps(result), encoding="utf-8")
            selected = regression.select_reference(root, root / "traces", self.contract)
            self.assertEqual(selected["trace_path"], target)
            self.assertEqual(selected["walker"], "walker-a")
            self.assertEqual(selected["candidate_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
