"""Local algebra/transport tests only; no simulator or real checkpoint acceptance."""
import ast
import contextlib
import io
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import modumorph_policy_adapter as binding
import modumorph_inference_worker as worker


def original_agent(full=False):
    """Execute original context algebra via AST, without importing MuJoCo."""
    tree = ast.parse((ROOT / "metamorph/envs/modules/agent.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Agent")
    methods = {"__init__", "get_context", "_select_obs"}
    if full:
        methods.update({"get_limb_obs", "get_joint_obs", "combine_limb_joint_obs", "_get_one_hot_body_idx"})
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    namespace = {"np": np, "cfg": SimpleNamespace(MIRROR_DATA_AUG=False, MODEL=SimpleNamespace(
        CONTEXT_OBS_TYPES=binding.CONTEXT_TYPES, PROPRIOCEPTIVE_OBS_TYPES=binding.PROPRIO_TYPES, MAX_LIMBS=3))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "original_agent_ast", "exec"), namespace)
    return namespace["Agent"]()


def fixture():
    rng = np.random.RandomState(71)
    model = SimpleNamespace(
        body_pos=rng.uniform(-.1, .1, (2, 3)), body_ipos=rng.uniform(-.1, .1, (2, 3)),
        body_iquat=np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
        geom_quat=np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
        body_mass=np.array([1.5, 3.]), geom_size=np.array([[.06, .1, 0.], [.08, .2, 0.]]),
        geom_friction=np.ones((2, 3)), jnt_pos=np.array([[0., 0., 0.], [.01, -.01, .02]]),
        jnt_range=np.array([[0., 0.], [-1., 1.]]), jnt_axis=np.array([[0., 0., 0.], [1., 0., 0.]]),
        actuator_gear=np.array([[120., 0., 0., 0., 0., 0.]]),
        dof_armature=np.zeros(7), dof_damping=np.zeros(7))
    agent = original_agent()
    agent.agent_body_idxs = np.arange(2)
    agent.agent_geom_idxs = np.arange(2)
    limb, joint = agent.get_context(SimpleNamespace(model=model))
    raw = np.zeros((3, 52))
    raw[:2, :13] = rng.randn(2, 13)
    raw[:2, 13:30] = np.hstack([model.body_pos, model.body_ipos, model.body_iquat,
                                model.geom_quat, model.body_mass[:, None], model.geom_size[:, :2]])
    raw[1, 30:41] = np.hstack([[[.3, .7]], model.jnt_pos[1:], model.jnt_range[1:],
                               model.jnt_axis[1:], model.actuator_gear[:, :1]])[0]
    mask = np.array([False, False, True])
    act_mask = np.array([True, True, False, True, True, True])
    expected = np.zeros((3, 35))
    expected[:2, :17] = limb
    expected[1, 17:26] = joint[0]
    return raw, mask, act_mask, expected


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.shape = self.value.shape
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value
    def tolist(self): return self.value.tolist()
    def to(self, device): return self
    def all(self): return self.value.all()


class BindingTests(unittest.TestCase):
    def test_worker_native_proprio_and_context_original_ast(self):
        raw, mask, act_mask, context = fixture()
        agent = original_agent(full=True)
        # Same nominal model used by the earlier independent raw feature fixture.
        rng = np.random.RandomState(71)
        model = dict(body_pos=rng.uniform(-.1, .1, (2, 3)), body_ipos=rng.uniform(-.1, .1, (2, 3)),
            body_iquat=np.tile([1., 0., 0., 0.], (2, 1)), geom_quat=np.tile([1., 0., 0., 0.], (2, 1)),
            body_mass=np.array([1.5, 3.]), geom_size=np.array([[.06, .1, 0.], [.08, .2, 0.]]),
            geom_friction=np.ones((2, 3)), jnt_pos=np.array([[0., 0., 0.], [.01, -.01, .02]]),
            jnt_range=np.array([[0., 0.], [-1., 1.]]), jnt_axis=np.array([[0., 0., 0.], [1., 0., 0.]]),
            actuator_gear=np.array([[120., 0., 0., 0., 0., 0.]]), dof_armature=np.zeros(7), dof_damping=np.zeros(7))
        data = dict(body_xpos=np.array([[2., 3., 4.], [5., 6., 7.]]),
            body_xvelp=np.arange(6).reshape(2, 3).astype(float), body_xvelr=np.arange(6, 12).reshape(2, 3).astype(float),
            body_xquat=np.tile([1., 0., 0., 0.], (2, 1)), qpos=np.r_[np.zeros(7), -.4], qvel=np.r_[np.zeros(6), .7])
        positions = data["body_xpos"].copy()
        positions[:, 0] -= 2.
        raw[:2, :13] = np.hstack([positions, data["body_xvelp"], data["body_xvelr"], data["body_xquat"]])
        # Match original division bit-for-bit, rather than decimal .3 rounding.
        raw[1, 30] = (data["qpos"][-1] + 1.) / 2.
        fake_module = SimpleNamespace(Agent=lambda: agent)
        with patch.dict(sys.modules, {"metamorph.envs.modules.agent": fake_module}):
            actual = worker.original_context({k: v.tolist() for k, v in model.items()},
                {k: v.tolist() for k, v in data.items()}, mask, act_mask, SimpleNamespace(MODEL=SimpleNamespace(MAX_LIMBS=3)))
        np.testing.assert_array_equal(actual["proprioceptive"], raw.ravel())
        np.testing.assert_array_equal(actual["context"], context.ravel())

    def test_context_exact_original_ast(self):
        raw, mask, act_mask, expected = fixture()
        np.testing.assert_array_equal(binding.static_context(raw.ravel(), mask, act_mask).reshape(3, 35), expected)

    def test_padded_slots_zero_and_dynamic_fields_ignored(self):
        raw, mask, act_mask, expected = fixture()
        raw[:2, :13] = 901.
        raw[:, 30:32] = 331.
        raw[:, 41:43] = -211.
        actual = binding.static_context(raw.ravel(), mask, act_mask).reshape(3, 35)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue((actual[2] == 0).all())
        self.assertTrue((actual[0, 17:] == 0).all())

    def test_invalid_context_shapes_and_values(self):
        raw, mask, act_mask, _ = fixture()
        for args in [(raw, mask, act_mask), (raw.ravel(), mask[:, None], act_mask),
                     (raw.ravel(), mask, act_mask[:-1]), (raw.ravel(), [0, 0, 2], act_mask)]:
            with self.assertRaises(ValueError): binding.static_context(*args)
        raw[0, 0] = np.nan
        with self.assertRaises(ValueError): binding.static_context(raw.ravel(), mask, act_mask)

    def test_allowlist_transport_and_mean_unchanged(self):
        adapter = binding.ModuMorphPolicyAdapter.__new__(binding.ModuMorphPolicyAdapter)
        adapter.max_limbs, adapter.device, adapter.walker = 3, "cpu", "w"
        adapter.bindings = {"w": np.arange(105, dtype=np.float32)}
        requests = []
        means = np.array([[4., -3., 2., 1., 0., -.5]], dtype=np.float32)
        adapter._request = lambda request: requests.append(request) or {"action": means.tolist()}
        obs = {"proprioceptive": Tensor(np.zeros((1, 156))),
               "obs_padding_mask": Tensor([[0, 0, 1]]),
               "act_padding_mask": Tensor([[1, 1, 0, 1, 1, 1]]), "edges": Tensor([[1, 0]])}
        for forbidden in ["history", "previous_action", "privileged_context", "dynamics", "context"]:
            obs[forbidden] = Tensor([[123456.]])
        fake_torch = SimpleNamespace(from_numpy=Tensor)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            _, distribution, _, _ = adapter(obs)
        np.testing.assert_array_equal(distribution.mean.numpy(), means)
        sent = requests[0]["obs"]
        self.assertEqual(set(sent), {"proprioceptive", "obs_padding_mask", "act_padding_mask", "edges", "context"})
        np.testing.assert_array_equal(sent["context"], adapter.bindings["w"][None])

    def test_worker_forwards_synthetic_distribution_mean(self):
        means = Tensor([[8., -9., 1., 2.]])
        calls = []
        def model(obs, compute_val):
            calls.append((obs, compute_val))
            return None, SimpleNamespace(mean=means), None, None, None, None
        model.parameters = lambda: []
        cfg = SimpleNamespace(MODEL=SimpleNamespace(MAX_LIMBS=2), RNG_SEED=1409)
        rms = {"proprioceptive": SimpleNamespace(mean=np.zeros(104), var=np.ones(104), count=7)}
        fake_torch = SimpleNamespace(float32="float32", inference_mode=contextlib.nullcontext,
            tensor=lambda value, **kwargs: Tensor(value), isfinite=lambda value: Tensor(np.isfinite(value.value)))
        output = io.StringIO()
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(worker, "load_policy", return_value=(model, rms, cfg)), \
             patch.object(sys, "argv", ["worker", "--checkpoint", "synthetic", "--config", "synthetic", "--device", "cpu"]), \
             patch.object(sys, "stdin", io.StringIO('{"op":"act","obs":{"context":[[0]]}}\n')), contextlib.redirect_stdout(output):
            worker.main()
        import json
        self.assertEqual(json.loads(output.getvalue().splitlines()[1])["action"], means.tolist())
        self.assertFalse(calls[0][1])

    def test_original_policy_provenance(self):
        source = (ROOT / "metamorph/algos/ppo/model.py").read_text()
        tree = ast.parse(source)
        agent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Agent")
        act = next(n for n in agent.body if isinstance(n, ast.FunctionDef) and n.name == "act")
        self.assertIn("act = pi.loc", ast.unparse(act))
        self.assertNotIn("history", ast.unparse(act))
        self.assertIn("HN", (ROOT / "README.md").read_text())


class OriginalCheckpointTests(unittest.TestCase):
    def setUp(self):
        import torch
        from metamorph.config import cfg
        from metamorph.algos.ppo.model import ActorCritic
        self.torch, self.cfg = torch, cfg
        self.saved_cfg = cfg.clone()
        cfg.MODEL.MAX_LIMBS = 3
        cfg.MODEL.MAX_JOINTS = 3
        cfg.MODEL.LIMB_EMBED_SIZE = 8
        cfg.MODEL.TRANSFORMER.NLAYERS = 1
        cfg.MODEL.TRANSFORMER.NHEAD = 2
        cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 16
        cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE = 8
        cfg.MODEL.TRANSFORMER.FIX_ATTENTION = True
        cfg.MODEL.TRANSFORMER.HYPERNET = True
        cfg.MODEL.TRANSFORMER.POS_EMBEDDING = None
        cfg.MODEL.TRANSFORMER.EMBEDDING_DROPOUT = False
        cfg.PPO.NUM_ENVS = 1
        torch.manual_seed(77)
        spaces = {"proprioceptive": SimpleNamespace(shape=(156,)), "context": SimpleNamespace(shape=(105,))}
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = ActorCritic(spaces, None)
        self.rms = {"proprioceptive": SimpleNamespace(mean=np.zeros(156), var=np.ones(156), count=8.)}
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="binding_test_", dir=ROOT / "tmp")
        self.cp, self.conf = Path(self.temp.name) / "synthetic.pt", Path(self.temp.name) / "config.yaml"
        self.conf.write_text(cfg.dump(), encoding="utf-8")
        torch.save([self.model, self.rms], self.cp)

    def tearDown(self):
        self.cfg.clear()
        self.cfg.update(self.saved_cfg)
        self.temp.cleanup()

    def load(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return worker.load_policy(self.cp, self.conf, "cpu")

    def test_full_object_load_eval_frozen_and_repeat_mean(self):
        model, rms, cfg = self.load()
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        torch = self.torch
        obs = {"proprioceptive": torch.zeros(1, 156), "context": torch.zeros(1, 105),
               "obs_padding_mask": torch.tensor([[0., 0., 1.]]),
               "act_padding_mask": torch.tensor([[1., 1., 0., 1., 1., 1.]]),
               "edges": torch.zeros(1, 6)}
        with torch.inference_mode():
            first = model(obs, compute_val=False)[1].mean
            second = model(obs, compute_val=False)[1].mean
        self.assertEqual(tuple(first.shape), (1, 6))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertFalse(first.requires_grad)

    def test_mismatched_serialized_architecture_rejected(self):
        import yaml
        data = yaml.safe_load(self.conf.read_text())
        data["MODEL"]["TRANSFORMER"]["LINEAR_CONTEXT_LAYER"] += 1
        self.conf.write_text(yaml.safe_dump(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "architecture settings"):
            self.load()

    def test_invalid_rms_rejected(self):
        for var, count in [(np.ones(155), 8.), (np.full(156, -1.), 8.),
                           (np.full(156, np.nan), 8.), (np.ones(156), 0.)]:
            self.rms["proprioceptive"].var, self.rms["proprioceptive"].count = var, count
            self.torch.save([self.model, self.rms], self.cp)
            with self.assertRaisesRegex(ValueError, "normalization statistics"):
                self.load()

    def test_parameter_signature_rejected(self):
        self.model.mu_net.register_parameter("unexpected_binding_test", self.torch.nn.Parameter(self.torch.zeros(1)))
        self.torch.save([self.model, self.rms], self.cp)
        with self.assertRaisesRegex(ValueError, "parameter names/shapes"):
            self.load()


class RunnerEvidenceTests(unittest.TestCase):
    def test_complete_evidence_tampering_rejected(self):
        import run_modumorph_frozen_eval as runner
        import json
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="resume_test_", dir=ROOT / "tmp") as directory:
            cell = Path(directory)
            target = cell / "results.json"
            target.write_text('{"schema_version":3}')
            (cell / "COMPLETE.json").write_text(json.dumps({"results.json": runner.sha(target)}))
            runner.verify_complete(cell)
            target.write_text('{"schema_version":999}')
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                runner.verify_complete(cell)

    def test_frozen_sources_seeds_and_schema_contract(self):
        import json
        import run_modumorph_frozen_eval as runner
        contract = json.loads((ROOT / "configs/modumorph_frozen_eval_binding.json").read_text())
        regression = json.loads((ROOT / "configs/modumorph_evaluator_behavior_regression.json").read_text())
        counterpart = ROOT.parent / "rmamorph"
        for name, digest in contract["source_sha256"].items():
            actual = runner.source_sha(counterpart / name)
            if name in regression["superseded_transient_hashes"]:
                self.assertNotEqual(actual, digest, name)
            else:
                self.assertEqual(actual, digest, name)
        self.assertFalse(regression["table2_authority_for_transient_hashes"])
        self.assertEqual(regression["candidate_authority_status"], "pending_behavior_regression")
        self.assertEqual(contract["formal_evaluation"]["episodes_per_walker"], 1)
        self.assertEqual(contract["evaluation_seeds"], [1409])
        tree = ast.parse((ROOT / "tools/run_modumorph_frozen_eval.py").read_text())
        source = ast.unparse(tree)
        self.assertIn("'schema_version': 3", source)
        self.assertIn("'training_seed': 'UNKNOWN'", source)
        self.assertIn("contract['evaluation_seeds']", source)

    def test_native_trace_fixture_validation(self):
        import json
        import run_modumorph_frozen_eval as runner
        raw = {"schema_version": 3, "protocols": {"ood_strong": {"per_walker": {"fixture": {
            "seeds": {"1409": {"returns": [1.], "lengths": [600], "adaptation": {"observed": True}}}}}}}}
        with tempfile.TemporaryDirectory(prefix="trace_schema_test_", dir=ROOT / "tmp") as directory:
            cell = Path(directory)
            traces = cell / "traces" / "fixture" / "1409"
            spec = importlib.util.spec_from_file_location("frozen_trace", ROOT.parent / "rmamorph/tools/morphadapt_trace.py")
            native = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(native)
            identity = {key: "fixture" for key in ("training_seed", "eval_seed", "method", "walker", "protocol", "setting", "checkpoint_sha256", "config_sha256", "walkers_file_sha256")}
            writer = native.EpisodeTraceWriter(traces, identity)
            for index in range(3):
                writer.record_step(reward=np.array([1.]), info={"measured_forward_velocity": 1., "recovery_metric_value": 1.},
                    policy_latent=None, policy_action=[.2, .3], action_history_used=None, mutation_step=2, done=index == 2)
            writer.finish_episode(episode={"r": 3., "l": 3, "dynamics/mid_episode_step": 2}, recovery=None, adaptation=None)
            writer.close()
            target = traces / "episodes.jsonl"
            episode = json.loads(target.read_text())
            runner.validate_smoke(cell, raw, {"mutation_step": 2}, "smoke")
            episode["mutation"]["step"] = 3
            target.write_text(json.dumps(episode) + "\n")
            with self.assertRaisesRegex(ValueError, "mutation step mismatch"):
                runner.validate_smoke(cell, raw, {"mutation_step": 2}, "smoke")
            episode["mutation"]["step"] = 2
            episode["steps"][-1]["terminated"] = False
            target.write_text(json.dumps(episode) + "\n")
            with self.assertRaisesRegex(ValueError, "completed episode"):
                runner.validate_smoke(cell, raw, {"mutation_step": 2}, "smoke")


if __name__ == "__main__":
    unittest.main()
