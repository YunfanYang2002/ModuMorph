"""Local contract tests. All weights/assets/simulator states here are MOCK fixtures."""
import ast
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import cloudpickle
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from metamorph.config import cfg
import modumorph_training_contract as contract
import run_modumorph_training as runner
import modumorph_inference_worker as inference
from metamorph.envs.vec_env import training_snapshot as snapshot
from metamorph.envs.vec_env.running_mean_std import RunningMeanStd


def function_ast(source, name):
    return next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == name)


def execute_function(node, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 "native_function_ast", "exec"), namespace)
    return namespace[node.name]


def immutable_native_source():
    fingerprint = json.loads((ROOT / "MODUMORPH_TRAINING_SEMANTIC_FINGERPRINT.json").read_text())
    return subprocess.check_output(["git", "show", fingerprint["source_git_sha"] + ":metamorph/algos/ppo/ppo.py"],
                                   cwd=ROOT, text=True)


class MockModel:
    nv = nu = nbody = ngeom = nsite = nsensordata = 1
    nmocap = nuserdata = 0

    def get_mjb(self):
        return b"MOCK-model-not-MuJoCo"


class MockSim:
    def __init__(self, model=None):
        self.model = model or MockModel()
        self.state = SimpleNamespace(time=1., qpos=np.array([.2]), qvel=np.array([.3]), act=np.array([.4]))
        self.data = SimpleNamespace(**{key: np.array([i + .5]) for i, key in enumerate(snapshot.FIELDS)})

    def get_state(self):
        return copy.deepcopy(self.state)

    def set_state(self, state):
        self.state = copy.deepcopy(state)

    def forward(self):
        # Deliberately overwrite integration fields: restore must put them back afterwards.
        for key in snapshot.FIELDS:
            getattr(self.data, key)[:] = -100


class MockEnv:
    def __init__(self):
        self.sim = MockSim()
        self.viewer, self._viewers = None, {}
        self.np_random = np.random.RandomState(1409)
        self.num_steps = 2560


class MockWrapper:
    def __init__(self):
        self._env = MockEnv()
        self._unimal_seq, self._unimal_seq_idx = [7, 3, 9], 1
        self._active_unimal_idx, self.num_steps = 3, 2560


class TrainingChecks(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="training_checks_", dir=ROOT / "tmp")
        self.out = Path(self.temporary.name)
        self.original = cfg.clone()
        torch.set_num_threads(1)

    def tearDown(self):
        cfg.clear()
        cfg.update(self.original)
        self.temporary.cleanup()

    def test_env_step_accounting_10m_prefix(self):
        result = contract.accounting(10_000_000)
        self.assertEqual(result["training_iterations"], 122)
        self.assertEqual(result["training_env_steps"], 9_994_240)
        self.assertEqual(result["unused_budget"], 5760)
        self.assertEqual(contract.accounting(100_000_000)["training_iterations"], 1220)
        with self.assertRaisesRegex(ValueError, "complete native rollout"):
            contract.accounting(81919)

    def test_workers_propagation_and_env_count_rejection(self):
        for workers in (1, 2, 4, 8, 16, 32):
            configured = contract.configure(self.out / str(workers), 10_000_000, workers)
            self.assertEqual(configured.PPO.NUM_ENVS, 32)
            self.assertEqual(configured.VECENV.IN_SERIES, 32 // workers)
            self.assertEqual(configured.PPO.MAX_ITERS, 1220)
            self.assertEqual(configured.PPO.EARLY_EXIT_MAX_ITERS, 122)
        for count in (16, 64):
            with self.assertRaisesRegex(ValueError, "NUM_ENVS must remain 32"):
                contract.configure(self.out, 10_000_000, 16, count)
        with self.assertRaisesRegex(ValueError, "workers must divide"):
            contract.configure(self.out, 10_000_000, 3)

    def test_original_seed_plus_lane_executed(self):
        source = (ROOT / "metamorph/algos/ppo/envs.py").read_text()
        made = []
        def create(*args, **kwargs):
            env = SimpleNamespace(seed=lambda value: made.append(value))
            return env
        namespace = {"gym": SimpleNamespace(make=create), "CUSTOM_ENVS": [],
                     "RecordEpisodeStatistics": lambda env: env}
        make = execute_function(function_ast(source, "make_env"), namespace)
        for lane in range(32):
            make("MOCK-v0", 1409, lane)()
        self.assertEqual(made, list(range(1409, 1441)))

    def test_worker_grouping_preserves_lane_order(self):
        # Execute actual constructor with MOCK multiprocessing, never start processes.
        source = (ROOT / "metamorph/envs/vec_env/subproc_vec_env.py").read_text()
        node = function_ast(source, "__init__")
        calls = []
        class Remote:
            def send(self, value): pass
            def recv(self): return SimpleNamespace(x=(None, None, "MOCK"))
            def close(self): pass
        class Process:
            def __init__(self, **kwargs): calls.append(kwargs["args"][2].x)
            def start(self): pass
        ctx = SimpleNamespace(Pipe=lambda: (Remote(), Remote()), Process=Process)
        namespace = {"np": np, "mp": SimpleNamespace(get_context=lambda _: ctx),
                     "worker": object(), "CloudpickleWrapper": lambda x: SimpleNamespace(x=x),
                     "clear_mpi_env_vars": contextlib.nullcontext,
                     "VecEnv": SimpleNamespace(__init__=lambda *args: None)}
        init = execute_function(node, namespace)
        for series in (1, 2, 4, 8, 16, 32):
            calls.clear()
            instance = SimpleNamespace()
            init(instance, list(range(32)), in_series=series)
            self.assertEqual(instance.nremotes, 32 // series)
            self.assertEqual([v for group in calls for v in group], list(range(32)))

    def split_fixture(self):
        binding = json.loads((ROOT / "configs/modumorph_training_binding.json").read_text())
        self.assertEqual(len(binding["walkers"]), 100)
        train = self.out / "authority/output/unimals_100/train"
        (train / "xml").mkdir(parents=True)
        (train / "metadata").mkdir()
        reference = train.parents[2] / binding["authority_config"]
        reference.parent.mkdir()
        reference.write_text(yaml.safe_dump({"ENV": {"WALKERS": binding["walkers"]}}, sort_keys=False))
        for name in binding["walkers"]:
            (train / "xml" / (name + ".xml")).write_text('<mujoco><worldbody><body name="torso/0"/></worldbody><actuator/><sensor/></mujoco>')
            (train / "metadata" / (name + ".json")).write_text('{"dof": 8, "num_limbs": 6}')
        return binding, train, reference

    def test_ordered_100_walker_binding(self):
        binding, train, reference = self.split_fixture()
        with patch.object(contract, "text_hash", return_value=binding["authority_config_sha256"]):
            result = contract.validate_split(train)
            self.assertEqual(result["walkers"], binding["walkers"])
            self.assertEqual(list(result["files"]), binding["walkers"])
            reference.write_text(yaml.safe_dump({"ENV": {"WALKERS": list(reversed(binding["walkers"]))}}))
            with self.assertRaisesRegex(ValueError, "walker order mismatch"):
                contract.validate_split(train)

    def test_split_rejects_asset_and_authority_mismatch(self):
        binding, train, reference = self.split_fixture()
        with self.assertRaisesRegex(ValueError, "config hash mismatch"):
            contract.validate_split(train)
        with patch.object(contract, "text_hash", return_value=binding["authority_config_sha256"]):
            (train / "metadata" / (binding["walkers"][0] + ".json")).unlink()
            with self.assertRaisesRegex(ValueError, "metadata IDs mismatch"):
                contract.validate_split(train)

    def test_strict_resume_provenance(self):
        saved = {"seed": 1409, "config_sha256": "config", "split": {"walkers": ["a", "b"], "files": {"a": "hash"}}}
        contract.resume_match(saved, copy.deepcopy(saved))
        for key, changed in (("seed", 1410), ("config_sha256", "other"), ("split", {"walkers": ["b", "a"]})):
            current = copy.deepcopy(saved)
            current[key] = changed
            with self.assertRaisesRegex(ValueError, key):
                contract.resume_match(saved, current)

    def test_no_algorithmic_config_drift(self):
        baseline = self.original.clone()
        baseline.merge_from_file(str(ROOT / "configs/ft.yaml"))
        baseline.merge_from_list(contract.RECIPE)
        actual = contract.configure(self.out, 10_000_000, 32).clone()
        # These are the explicitly permitted delivery/split/budget/grouping fields.
        for node, fields in (("", ("OUT_DIR", "RNG_SEED")),
                             ("ENV", ("WALKER_DIR", "WALKERS")),
                             ("MODEL", ("MAX_LIMBS", "MAX_JOINTS")),
                             ("VECENV", ("IN_SERIES",)),
                             ("PPO", ("MAX_ITERS", "EARLY_EXIT", "EARLY_EXIT_STATE_ACTION_PAIRS", "EARLY_EXIT_MAX_ITERS"))):
            target = actual[node] if node else actual
            origin = baseline[node] if node else baseline
            for field in fields:
                target[field] = origin[field]
        self.assertEqual(actual.dump(), baseline.dump())

    def test_nominal_no_dr_modules(self):
        configured = contract.configure(self.out, 10_000_000, 16)
        self.assertEqual(configured.ENV.MODULES, ["Agent", "Floor"])
        self.assertFalse(any("random" in key.lower() or key == "DR" for key in configured.ENV))
        source = (ROOT / "metamorph/envs/tasks/unimal.py").read_text()
        update = function_ast(source, "update")
        self.assertEqual({n.attr for n in ast.walk(update) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)},
                         {"unimal_id", "unimal_idx", "xml_str"})

    def test_tmp_guard_path_escape(self):
        self.assertEqual(contract.output_path(self.out / "ok"), self.out / "ok")
        for path in (ROOT, ROOT / "tmp", self.out / "../../outside", ROOT / "tmp-other/result"):
            with self.assertRaisesRegex(ValueError, "project ./tmp"):
                contract.output_path(path)

    def test_tmp_guard_symlink_escape(self):
        # Both link and destination fixture stay under project tmp; destination is
        # outside a mocked project boundary, so no external filesystem mutation.
        fake_root = self.out / "project"
        (fake_root / "tmp").mkdir(parents=True)
        external = self.out / "outside"
        external.mkdir()
        link = fake_root / "tmp/link"
        try:
            link.symlink_to(external, target_is_directory=True)
        except OSError as error:
            self.skipTest("Windows symlink privilege unavailable: " + str(error))
        with patch.object(contract, "ROOT", fake_root):
            with self.assertRaisesRegex(ValueError, "project ./tmp"):
                contract.output_path(link / "result")

    def test_mock_env_snapshot_restores_rng_and_integration(self):
        env = MockWrapper()
        state = snapshot.capture_worker([env])
        expected_python, expected_numpy = random.random(), np.random.rand()
        expected_private = env._env.np_random.rand()
        fake = ModuleType("mujoco_py")
        fake.MjSim, fake.load_model_from_mjb = MockSim, lambda path: MockModel()
        with patch.dict(sys.modules, {"mujoco_py": fake}):
            restored = snapshot.restore_worker(state, self.out / "scratch")[0]
        self.assertEqual(restored._unimal_seq, [7, 3, 9])
        self.assertEqual(restored._unimal_seq_idx, 1)
        self.assertEqual(restored.num_steps, 2560)
        self.assertEqual(restored._env.np_random.rand(), expected_private)
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.rand(), expected_numpy)
        for field in snapshot.FIELDS:
            np.testing.assert_array_equal(getattr(restored._env.sim.data, field), getattr(env._env.sim.data, field))

    def test_snapshot_rejects_live_viewer(self):
        env = MockEnv()
        env.viewer = object()
        with self.assertRaisesRegex(ValueError, "active render viewers"):
            snapshot.capture(env)

    def test_mjb_restore_passes_exact_bytes_from_nonempty_file(self):
        env = MockEnv()
        mjb = b"\x00MOCK-MJB\xff\x80\x00"
        with patch.object(env.sim.model, "get_mjb", return_value=mjb):
            saved = snapshot.capture(env)
        self.assertEqual(saved["mjb"], mjb)
        scratch = self.out / "scratch"
        arguments = []
        def load(model_bytes):
            self.assertIs(type(model_bytes), bytes)
            self.assertEqual(model_bytes, mjb)
            paths = list(scratch.glob("*.mjb"))
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].is_file())
            self.assertGreater(paths[0].stat().st_size, 0)
            self.assertEqual(paths[0].read_bytes(), mjb)
            arguments.append(model_bytes)
            return MockModel()
        fake = ModuleType("mujoco_py")
        fake.MjSim, fake.load_model_from_mjb = MockSim, load
        with patch.dict(sys.modules, {"mujoco_py": fake}):
            restored = snapshot.restore(saved, scratch)
        self.assertEqual(arguments, [mjb])
        np.testing.assert_array_equal(restored.sim.state.qpos, env.sim.state.qpos)
        self.assertEqual(list(scratch.glob("*.mjb")), [])

    def test_mjb_capture_rejects_empty_bytes(self):
        env = MockEnv()
        with patch.object(env.sim.model, "get_mjb", return_value=b""):
            with self.assertRaisesRegex(ValueError, "non-empty bytes"):
                snapshot.capture(env)

    def test_mjb_restore_rejects_missing_and_empty_file_before_loader(self):
        saved = snapshot.capture(MockEnv())
        original = tempfile.NamedTemporaryFile
        for missing in (True, False):
            with self.subTest(missing=missing):
                @contextlib.contextmanager
                def altered_file(**kwargs):
                    with original(**kwargs) as stream:
                        yield stream
                    path = Path(stream.name)
                    if missing:
                        path.unlink()
                    else:
                        path.write_bytes(b"")
                fake = ModuleType("mujoco_py")
                fake.MjSim = MockSim
                fake.load_model_from_mjb = unittest.mock.Mock()
                with patch.dict(sys.modules, {"mujoco_py": fake}), patch.object(snapshot.tempfile, "NamedTemporaryFile", altered_file):
                    expected = FileNotFoundError if missing else ValueError
                    with self.assertRaises(expected):
                        snapshot.restore(saved, self.out / "scratch")
                fake.load_model_from_mjb.assert_not_called()
                self.assertEqual(list((self.out / "scratch").glob("*.mjb")), [])

    def optional_sim(self):
        sim = MockSim()
        for key in ("mocap_pos", "mocap_quat", "userdata"):
            setattr(sim.data, key, None)
        return sim

    def test_snapshot_none_capture(self):
        sim = self.optional_sim()
        integration = snapshot.capture_integration(sim)
        self.assertEqual({key for key, value in integration.items() if value is None},
                         {"mocap_pos", "mocap_quat", "userdata"})
        env = MockEnv()
        env.sim = sim
        self.assertIsNone(snapshot.capture(env)["integration"]["userdata"])

    def test_snapshot_array_independent_copy(self):
        sim = self.optional_sim()
        sim.data.ctrl = np.arange(4., dtype=np.float64).reshape(2, 2)
        saved = snapshot.capture_integration(sim)["ctrl"]
        self.assertFalse(np.shares_memory(saved, sim.data.ctrl))
        sim.data.ctrl[0, 0] = 99.
        self.assertEqual(saved[0, 0], 0.)
        saved[1, 1] = -20.
        self.assertEqual(sim.data.ctrl[1, 1], 3.)

    def test_snapshot_scalar_capture(self):
        sim = self.optional_sim()
        sim.data.ctrl = 1.25
        sim.data.qacc = np.float32(2.5)
        integration = snapshot.capture_integration(sim)
        self.assertEqual(integration["ctrl"], 1.25)
        self.assertIs(type(integration["ctrl"]), float)
        self.assertIs(type(integration["qacc"]), np.float32)

    def test_restore_none_to_none(self):
        sim = self.optional_sim()
        saved = snapshot.capture_integration(sim)
        snapshot.restore_integration(sim, saved)
        for key in ("mocap_pos", "mocap_quat", "userdata"):
            self.assertIsNone(getattr(sim.data, key))

    def test_restore_array_exact(self):
        sim = self.optional_sim()
        sim.data.ctrl = np.array([[1.5, -0.0], [3.25, 4.]], dtype=np.float32)
        saved = snapshot.capture_integration(sim)
        sim.data.ctrl[:] = 99.
        snapshot.restore_integration(sim, saved)
        self.assertEqual(sim.data.ctrl.dtype, np.dtype("float32"))
        self.assertEqual(sim.data.ctrl.tobytes(), saved["ctrl"].tobytes())

    def test_restore_scalar_exact(self):
        sim = self.optional_sim()
        sim.data.ctrl, sim.data.qacc = 1.25, np.float32(2.5)
        saved = snapshot.capture_integration(sim)
        sim.data.ctrl, sim.data.qacc = -9., np.float32(-4.)
        snapshot.restore_integration(sim, saved)
        self.assertEqual(sim.data.ctrl, 1.25)
        self.assertIs(type(sim.data.ctrl), float)
        self.assertEqual(sim.data.qacc.tobytes(), saved["qacc"].tobytes())

    def test_restore_none_to_value_rejected(self):
        sim = self.optional_sim()
        saved = snapshot.capture_integration(sim)
        sim.data.userdata = np.zeros(0)
        with self.assertRaisesRegex(ValueError, "userdata None/value"):
            snapshot.restore_integration(sim, saved)

    def test_restore_value_to_none_rejected(self):
        sim = self.optional_sim()
        sim.data.userdata = np.zeros(0)
        saved = snapshot.capture_integration(sim)
        sim.data.userdata = None
        with self.assertRaisesRegex(ValueError, "userdata None/value"):
            snapshot.restore_integration(sim, saved)

    def test_restore_shape_mismatch_rejected(self):
        sim = self.optional_sim()
        saved = snapshot.capture_integration(sim)
        sim.data.ctrl = np.zeros((1, 1))
        with self.assertRaisesRegex(ValueError, "ctrl shape mismatch"):
            snapshot.restore_integration(sim, saved)

    def test_restore_dtype_and_representation_mismatch_rejected(self):
        sim = self.optional_sim()
        saved = snapshot.capture_integration(sim)
        sim.data.ctrl = np.zeros(1, dtype=np.float32)
        with self.assertRaisesRegex(TypeError, "ctrl dtype mismatch"):
            snapshot.restore_integration(sim, saved)
        sim.data.ctrl = [0.]
        with self.assertRaisesRegex(TypeError, "ctrl representation mismatch"):
            snapshot.restore_integration(sim, saved)
        sim.data.ctrl = 1.25
        saved = snapshot.capture_integration(sim)
        sim.data.ctrl = np.float64(0.)
        with self.assertRaisesRegex(TypeError, "ctrl representation mismatch"):
            snapshot.restore_integration(sim, saved)

    def test_unexpected_none_and_missing_field_rejected(self):
        sim = self.optional_sim()
        sim.data.ctrl = None
        with self.assertRaisesRegex(ValueError, "ctrl.*model.nu=1"):
            snapshot.capture_integration(sim)
        integration = {key: getattr(sim.data, key) for key in snapshot.FIELDS}
        with self.assertRaisesRegex(ValueError, "ctrl.*model.nu=1"):
            snapshot.restore_integration(sim, integration)
        del sim.data.ctrl
        with self.assertRaisesRegex(AttributeError, "ctrl"):
            snapshot.capture_integration(sim)

    def test_none_fields_diagnostic_once(self):
        sim = self.optional_sim()
        output = io.StringIO()
        with patch.object(snapshot, "_reported_none_fields", set()), contextlib.redirect_stdout(output):
            snapshot.capture_integration(sim)
            snapshot.capture_integration(sim)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0].split("=", 1)[1]), ["mocap_pos", "mocap_quat", "userdata"])

    def test_array_like_and_zero_dimensional_restore(self):
        sim = self.optional_sim()
        sim.data.ctrl, sim.data.qacc = [[1., 2.]], np.array(3.5)
        saved = snapshot.capture_integration(sim)
        sim.data.ctrl[0][0], sim.data.qacc[...] = 9., 10.
        self.assertEqual(saved["ctrl"], [[1., 2.]])
        snapshot.restore_integration(sim, saved)
        self.assertEqual(sim.data.ctrl, [[1., 2.]])
        np.testing.assert_array_equal(sim.data.qacc, np.array(3.5))

    def test_none_training_digest_is_stable_and_distinguishes_array(self):
        env = MockEnv()
        env.sim = self.optional_sim()
        env.sim.state.act = None
        vector = {"workers": [{"envs": [snapshot.capture(env)]}],
                  "ob_rms": {"proprioceptive": RunningMeanStd(shape=(1,))},
                  "ret_rms": RunningMeanStd(), "ret": np.zeros(1)}
        model = torch.nn.Linear(1, 1)
        trainer = SimpleNamespace(actor_critic=model, optimizer=torch.optim.Adam(model.parameters()),
                                  train_meter=SimpleNamespace(agent_meters={}))
        observer = SimpleNamespace(out=self.out, updates=0)
        (self.out / "sampling.json").write_text('[1]')
        original_asarray = np.asarray
        def numeric_array(value, *args, **kwargs):
            self.assertIsNotNone(value, "None must not be hashed as an object-array memory address")
            return original_asarray(value, *args, **kwargs)
        with patch.object(runner, "vector_capture", return_value=vector), patch.object(np, "asarray", side_effect=numeric_array):
            expected = runner.training_digest(trainer, observer)
            self.assertEqual(runner.training_digest(trainer, observer), expected)
            vector["workers"][0]["envs"][0]["integration"]["userdata"] = np.zeros(0)
            self.assertNotEqual(runner.training_digest(trainer, observer), expected)

    def test_benchmark_rerun_preserves_completed_all_failed_evidence(self):
        out = self.out / "benchmark"
        out.mkdir()
        (out / "results.json").write_text('[]')
        (out / "hardware.json").write_text('{"candidates": [1, 4, 8, 16, 32]}')
        for workers in (1, 4, 8, 16, 32):
            (out / f"workers_{workers}_failure.json").write_text('{"status": "FAIL"}')
        (out / "original.log").write_text('original None.copy traceback')
        with patch.object(runner.subprocess, "Popen", side_effect=RuntimeError("MOCK_STOP")), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "MOCK_STOP"):
                runner.benchmark(SimpleNamespace(output=str(out)), {"candidates": [16]})
        archives = list(self.out.glob('benchmark_failed_*'))
        self.assertEqual(len(archives), 1)
        self.assertEqual((archives[0] / "original.log").read_text(), 'original None.copy traceback')
        self.assertTrue(out.is_dir())

    def test_benchmark_rerun_rejects_successful_or_partial_evidence(self):
        out = self.out / "benchmark"
        out.mkdir()
        (out / "results.json").write_text('[{"status":"PASS"}]')
        with self.assertRaisesRegex(FileExistsError, "successful evidence"):
            runner.benchmark(SimpleNamespace(output=str(out)), {"candidates": [16]})
        (out / "results.json").write_text('[]')
        (out / "hardware.json").write_text('{"candidates": [16]}')
        with self.assertRaisesRegex(FileExistsError, "not a completed all-failed attempt"):
            runner.benchmark(SimpleNamespace(output=str(out)), {"candidates": [16]})
        self.assertEqual(list(self.out.glob('benchmark_failed_*')), [])

    def test_original_actorcritic_and_rms_inference_load(self):
        from metamorph.algos.ppo.model import ActorCritic
        configured = contract.configure(self.out, 10_000_000, 16)
        configured.DEVICE = "cpu"
        spaces = {"proprioceptive": SimpleNamespace(shape=(12 * 52,)), "context": SimpleNamespace(shape=(12 * 35,))}
        with contextlib.redirect_stdout(io.StringIO()):
            model = ActorCritic(spaces, None)
        rms = {"proprioceptive": RunningMeanStd(shape=(12 * 52,))}
        rms["proprioceptive"].update(np.arange(2 * 12 * 52).reshape(2, -1) / 100)
        path, config_path = self.out / "MOCK-original-policy.pt", self.out / "config.yaml"
        torch.save([model, rms], path)
        config_path.write_text(configured.dump())
        with contextlib.redirect_stdout(io.StringIO()):
            loaded, stats, loaded_cfg = inference.load_policy(path, config_path, "cpu")
        self.assertIsInstance(loaded, ActorCritic)
        self.assertFalse(loaded.training)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]))
        np.testing.assert_array_equal(stats["proprioceptive"].mean, rms["proprioceptive"].mean)
        self.assertEqual(loaded_cfg.PPO.NUM_ENVS, 1)

    def test_reduced_budget_mock_checkpoint_metadata(self):
        for folder in ("checkpoints", "status", "logs"):
            (self.out / folder).mkdir()
        (self.out / "sampling.json").write_text('[1, 2, 3]')
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        normalized = SimpleNamespace(ob_rms={"proprioceptive": RunningMeanStd(shape=(2,))},
                                     ret_rms=RunningMeanStd(), ret=np.array([.3, .4]),
                                     venv=SimpleNamespace(capture_training_state=lambda: {"MOCK": True}))
        trainer = SimpleNamespace(actor_critic=model, optimizer=optimizer,
                                  train_meter=SimpleNamespace(MOCK=True), envs=SimpleNamespace(venv=normalized))
        signature = {"git_sha": "MOCK", "config_sha256": "MOCK"}
        observer = runner.PilotObserver(self.out, signature, 10_000_000, False)
        observer.iterations, observer.steps, observer.updates = 122, 9_994_240, 100
        # Avoid importing simulator-containing env factory; the helper's AST is
        # exercised elsewhere through native normalization and policy tests.
        fake_envs = ModuleType("metamorph.algos.ppo.envs")
        fake_envs.get_ob_rms = lambda envs: envs.venv.ob_rms
        with patch.dict(sys.modules, {"metamorph.algos.ppo.envs": fake_envs}):
            observer.save(trainer, {"proprioceptive": torch.ones(2)}, 1.)
        pointer = json.loads((self.out / "status/last_valid.json").read_text())
        directory = self.out / pointer["path"]
        metadata = json.loads((directory / "metadata.json").read_text())
        self.assertTrue(metadata["reduced_budget"])
        self.assertFalse(metadata["formal_matched_baseline"])
        self.assertEqual(metadata["training_env_steps"], 9_994_240)
        self.assertEqual(contract.sha256(directory / "metadata.json"), pointer["metadata_sha256"])
        payload = torch.load(directory / "resume.pt", weights_only=False)
        self.assertEqual(payload["vector"]["workers"], {"MOCK": True})
        self.assertEqual(payload["sampling"], b'[1, 2, 3]')
        self.assertTrue(torch.equal(payload["model"]["weight"], model.weight))

    def test_observer_disabled_loss_matches_git_baseline(self):
        baseline = immutable_native_source()
        current = (ROOT / "metamorph/algos/ppo/ppo.py").read_text()
        original = function_ast(baseline, "train_on_batch")
        modified = function_ast(current, "train_on_batch")
        class RemoveObserver(ast.NodeTransformer):
            def visit_If(self, node):
                if "training_observer" in ast.unparse(node.test):
                    return None
                return self.generic_visit(node)
        modified = RemoveObserver().visit(modified)
        self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(modified, include_attributes=False))
        # Execute both actual native loss bodies against the same deterministic
        # tiny model/buffer, no original PPO constructor or simulator import.
        def run(node, observer=None):
            torch.manual_seed(4)
            model = torch.nn.Linear(2, 1)
            model.log_std = torch.nn.Parameter(torch.zeros(1))
            class CallableModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = model
                def forward(self, obs, act, **kwargs):
                    val = self.net(obs)
                    return val, None, val * .1, val.mean() * 0 + .5, None, None
            actor = CallableModel()
            observations = torch.tensor([[.2, .3], [.4, .1]])
            batch = {"obs": observations, "act": torch.zeros(2, 1), "logp_old": torch.zeros(2, 1),
                     "adv": torch.tensor([[.2], [-.1]]), "ret": torch.ones(2, 1), "val": torch.zeros(2, 1),
                     "dropout_mask_v": None, "dropout_mask_mu": None, "unimal_ids": [0, 0]}
            trainer = SimpleNamespace(actor_critic=actor, optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
                                      buffer=SimpleNamespace(ret=torch.tensor([[.3], [.8]]), val=torch.zeros(2, 1),
                                                             get_sampler=lambda _: [batch]),
                                      train_meter=SimpleNamespace(add_train_stat=lambda *args: None),
                                      log_std_param="net.log_std", writer=SimpleNamespace(add_histogram=lambda *args: None))
            trainer.training_observer = observer
            local_cfg = cfg.clone()
            local_cfg.PPO.EPOCHS = 2
            function = execute_function(copy.deepcopy(node), {"torch": torch, "np": np, "nn": torch.nn, "cfg": local_cfg})
            function(trainer, 1)
            return actor.state_dict(), trainer.optimizer.state_dict()
        expected, expected_optimizer = run(original)
        actual, actual_optimizer = run(function_ast(current, "train_on_batch"))
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]), key)
        for index, values in expected_optimizer["state"].items():
            for key, value in values.items():
                self.assertTrue(torch.equal(value, actual_optimizer["state"][index][key]))
        counts = {"batch": 0, "optimizer": 0}
        observer = SimpleNamespace(on_batch=lambda *args: counts.__setitem__("batch", counts["batch"] + 1),
                                   after_optimizer_step=lambda trainer: counts.__setitem__("optimizer", counts["optimizer"] + 1))
        enabled, enabled_optimizer = run(function_ast(current, "train_on_batch"), observer)
        self.assertEqual(counts, {"batch": 2, "optimizer": 2})
        for key in actual:
            self.assertTrue(torch.equal(actual[key], enabled[key]), key)
        for index, values in actual_optimizer["state"].items():
            for key, value in values.items():
                self.assertTrue(torch.equal(value, enabled_optimizer["state"][index][key]))

    def test_observer_disabled_native_collection_matches_immutable_baseline(self):
        def run(source):
            events = []
            local_cfg = cfg.clone()
            local_cfg.PPO.NUM_ENVS, local_cfg.PPO.TIMESTEPS = 2, 3
            local_cfg.PPO.MAX_ITERS, local_cfg.PPO.EARLY_EXIT = 102, True
            local_cfg.PPO.EARLY_EXIT_MAX_ITERS = 101
            local_cfg.MODEL.TRANSFORMER.PER_NODE_EMBED = True
            local_cfg.LOG_PERIOD = -1
            optimizer = SimpleNamespace(param_groups=[{"lr": 0.}])
            class Env:
                def __init__(self): self.count = 0
                def reset(self):
                    events.append(("reset",))
                    return {"MOCK": torch.zeros(2, 1)}
                def get_unimal_idx(self):
                    ids = [self.count % 5, (self.count + 1) % 5]
                    events.append(("morphology", tuple(ids)))
                    return ids
                def step(self, action):
                    self.count += 1
                    events.append(("step", self.count, tuple(action.flatten().tolist())))
                    return {"MOCK": torch.full((2, 1), float(self.count))}, torch.ones(2, 1), [self.count % 2 == 0, False], [{"timeout": True}, {}]
            def act(obs, unimal_ids):
                events.append(("act", tuple(unimal_ids), tuple(obs["MOCK"].flatten().tolist())))
                return torch.zeros(2, 1), obs["MOCK"] + .1, torch.zeros(2, 1), None, None
            def insert(obs, action, logp, val, reward, masks, timeouts, *args):
                events.append(("insert", tuple(obs["MOCK"].flatten().tolist()), tuple(masks.flatten().tolist()),
                               tuple(timeouts.flatten().tolist()), tuple(args[-1])))
            def get_lr(iteration):
                value = 3e-4 * .5 * (1 + math.cos(math.pi * iteration / local_cfg.PPO.MAX_ITERS))
                events.append(("lr", iteration, value))
                return value
            def set_lr(target, value, scale):
                target.param_groups[0]["lr"] = value * scale[0]
            trainer = SimpleNamespace(envs=Env(), device="cpu", optimizer=optimizer, lr_scale=[1.],
                                      agent=SimpleNamespace(act=act, get_value=lambda obs, unimal_ids: torch.ones(2, 1)),
                                      buffer=SimpleNamespace(to=lambda device: events.append(("buffer_to", device)),
                                                             insert=insert, compute_returns=lambda value: events.append(("returns",))),
                                      train_meter=SimpleNamespace(add_ep_info=lambda infos: None, update_mean=lambda: None,
                                                                  mean_ep_rews={"reward": []}),
                                      train_on_batch=lambda iteration: events.append(("optimizer_trigger", iteration, optimizer.param_groups[0]["lr"])),
                                      save_sampled_agent_seq=lambda iteration: events.append(("sampling", iteration)),
                                      save_model=lambda iteration: events.append(("checkpoint", iteration)), file_prefix="MOCK")
            function = execute_function(function_ast(source, "train"), {"cfg": local_cfg, "torch": torch, "time": time,
                                                                         "ou": SimpleNamespace(get_iter_lr=get_lr, set_lr=set_lr)})
            with contextlib.redirect_stdout(io.StringIO()):
                function(trainer)
            return events
        expected = run(immutable_native_source())
        actual = run((ROOT / "metamorph/algos/ppo/ppo.py").read_text())
        self.assertEqual(actual, expected)
        self.assertEqual(len([event for event in actual if event[0] == "step"]) * 2, 606)
        self.assertEqual([event[1] for event in actual if event[0] == "sampling"], [0] + list(range(101)))
        self.assertEqual([event[1] for event in actual if event[0] == "checkpoint"], [0, 100])
        self.assertEqual([event[1] for event in actual if event[0] == "optimizer_trigger"], list(range(101)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
