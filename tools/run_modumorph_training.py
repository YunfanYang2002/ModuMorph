"""Explicit server-only throughput benchmark and reduced-budget pilot."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from modumorph_training_contract import accounting, configure, output_path, resume_match, sha256, text_hash, validate_split


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_signature():
    paths = ["configs/ft.yaml", "metamorph/config.py", "metamorph/algos/ppo/ppo.py",
             "metamorph/algos/ppo/buffer.py", "metamorph/algos/ppo/envs.py",
             "metamorph/algos/ppo/model.py", "metamorph/algos/ppo/transformer.py",
             "metamorph/utils/optimizer.py", "metamorph/envs/vec_env/subproc_vec_env.py",
             "metamorph/envs/vec_env/training_snapshot.py", "tools/modumorph_training_contract.py",
             "tools/run_modumorph_training.py", "configs/modumorph_training_binding.json"]
    # Include native physics, wrappers, normalization and sampler helpers.
    paths += [str(p.relative_to(ROOT)).replace("\\", "/") for directory in
              (ROOT / "metamorph/envs", ROOT / "metamorph/utils", ROOT / "modular")
              for p in directory.rglob("*") if p.suffix in (".py", ".xml")]
    return {p: text_hash(ROOT / p) for p in sorted(set(paths))}


def preflight():
    import psutil
    import torch
    print("PYTHON=" + sys.executable)
    if sys.platform != "linux" or not torch.cuda.is_available():
        raise RuntimeError("native fork + CUDA training requires Linux and an available CUDA GPU")
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    if not visible.isdigit():
        raise ValueError("select exactly one physical GPU ID")
    affinity = len(os.sched_getaffinity(0))
    print("CPU_COUNT=" + str(psutil.cpu_count()))
    print("CPU_AFFINITY_COUNT=" + str(affinity))
    print("AVAILABLE_RAM_BYTES=" + str(psutil.virtual_memory().available))
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu", "--format=csv"], check=True)
    subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv"], check=True)
    candidates = sorted(set([16] + [n for n in (1, 4, 8, 16, 32) if n <= affinity]))
    print("NUM_ENVS=32 (other counts rejected to preserve sample semantics)")
    print("CANDIDATE_CPU_WORKERS=" + json.dumps(candidates))
    print("THREAD_ENV=" + json.dumps({k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}))
    return {"cpu_count": psutil.cpu_count(), "cpu_affinity_count": affinity,
            "available_ram_bytes": psutil.virtual_memory().available, "candidates": candidates,
            "gpu": visible, "python": sys.executable}


class Resources:
    def __init__(self, out):
        import psutil
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.samples = []
        self.out = out
        self.failure = None
        self.executor = ThreadPoolExecutor(max_workers=1)

    def collect(self):
        import psutil
        while not self.stop.is_set():
            processes = [self.process] + self.process.children(recursive=True)
            rss = cpu_seconds = 0
            for process in processes:
                try:
                    rss += process.memory_info().rss
                    timings = process.cpu_times()
                    cpu_seconds += timings.user + timings.system
                except psutil.NoSuchProcess:
                    continue  # genuine process inventory race, not a training fallback
            result = subprocess.run(["nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
                                     "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode:
                self.failure = result.stderr
                return
            gpu, memory = map(float, result.stdout.strip().split(","))
            self.samples.append({"time": time.monotonic(), "cpu_seconds": cpu_seconds,
                                 "rss_bytes": rss, "gpu_utilization_percent": gpu, "gpu_memory_mib": memory})
            self.stop.wait(1)

    def finish(self):
        self.stop.set()
        self.future.result()
        self.executor.shutdown()
        write_json(self.out / "logs/resources.json", self.samples)
        if self.failure:
            raise RuntimeError("resource telemetry failed: " + str(self.failure))
        if len(self.samples) < 2:
            raise RuntimeError("insufficient resource telemetry")
        first, last = self.samples[0], self.samples[-1]
        return {"cpu_utilization_percent": 100 * (last["cpu_seconds"]-first["cpu_seconds"])/(last["time"]-first["time"]),
                "rss_peak_bytes": max(s["rss_bytes"] for s in self.samples),
                "gpu_utilization_mean_percent": sum(s["gpu_utilization_percent"] for s in self.samples)/len(self.samples),
                "gpu_memory_peak_mib": max(s["gpu_memory_mib"] for s in self.samples)}


def vector_capture(trainer):
    normalized = trainer.envs.venv
    return {"workers": normalized.venv.capture_training_state(),
            "ob_rms": copy.deepcopy(normalized.ob_rms), "ret_rms": copy.deepcopy(normalized.ret_rms),
            "ret": normalized.ret.copy()}


def vector_restore(trainer, state):
    from metamorph.config import cfg
    normalized = trainer.envs.venv
    normalized.venv.restore_training_state(state["workers"], Path(cfg.OUT_DIR) / "scratch")
    normalized.ob_rms = copy.deepcopy(state["ob_rms"])
    normalized.ret_rms = copy.deepcopy(state["ret_rms"])
    normalized.ret = state["ret"].copy()


def training_digest(trainer, observer):
    """Compare numerical training state across worker groupings, excluding wall time."""
    import hashlib
    import cloudpickle
    import numpy as np
    digest = hashlib.sha256()
    def array(value):
        digest.update(np.asarray(value).tobytes())
    for name, tensor in sorted(trainer.actor_critic.state_dict().items()):
        digest.update(name.encode())
        array(tensor.detach().cpu().numpy())
    for index, values in sorted(trainer.optimizer.state_dict()["state"].items()):
        digest.update(str(index).encode())
        for key, value in sorted(values.items()):
            digest.update(key.encode())
            array(value.detach().cpu().numpy() if hasattr(value, "detach") else value)
    state = vector_capture(trainer)
    for rms in list(state["ob_rms"].values()) + [state["ret_rms"]]:
        array(rms.mean)
        array(rms.var)
        array(rms.count)
    array(state["ret"])
    for worker in state["workers"]:
        for env in worker["envs"]:
            for cls, fields, link in cloudpickle.loads(env["layers"]):
                digest.update(cls.__name__.encode())
                for key in ("_unimal_seq", "_unimal_seq_idx", "_active_unimal_idx", "num_steps",
                            "unimal_id", "unimal_idx", "step_count", "_elapsed_steps",
                            "episode_return", "episode_length", "np_random"):
                    if key in fields:
                        digest.update(key.encode())
                        digest.update(cloudpickle.dumps(fields[key]))
            sim = env["sim_state"]
            for value in (sim.time, sim.qpos, sim.qvel, sim.act):
                array(value)
            for key in sorted(env["integration"]):
                array(env["integration"][key])
    for name, meter in sorted(trainer.train_meter.agent_meters.items()):
        digest.update(name.encode())
        array(meter.ep_len)
        array(meter.ep_len_ema)
        array(meter.ep_count)
    digest.update((observer.out / "sampling.json").read_bytes())
    digest.update(str(observer.updates).encode())
    return digest.hexdigest()


def equality(actual, expected):
    import numpy as np
    import torch
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            if key != "t":  # wall-clock episode time is not trajectory state
                equality(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equality(a, b)
    elif isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected), "snapshot replay tensor mismatch"
    elif isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected, "snapshot replay scalar mismatch"


class PilotObserver:
    def __init__(self, out, signature, budget, resume):
        self.out, self.signature, self.budget, self.resume = out, signature, budget, resume
        self.iterations = self.steps = self.updates = 0
        self.nan_count = self.numerical_failures = 0
        self.elapsed_before = 0
        self.entropy = []
        self.started = time.monotonic()

    def initialize(self, trainer):
        import cloudpickle
        import torch
        if self.resume:
            pointer = json.loads((self.out / "status/last_valid.json").read_text())
            checkpoint = output_path(self.out / pointer["path"])
            if checkpoint.parent != self.out / "checkpoints" or sha256(checkpoint / "metadata.json") != pointer["metadata_sha256"]:
                raise ValueError("invalid last-valid checkpoint pointer or metadata checksum")
            metadata = json.loads((checkpoint / "metadata.json").read_text())
            resume_match(metadata["signature"], self.signature)
            for name, digest in metadata["hashes"].items():
                if sha256(checkpoint / name) != digest:
                    raise ValueError("checkpoint checksum mismatch: " + name)
            payload = torch.load(checkpoint / "resume.pt", map_location=trainer.device, weights_only=False)
            trainer.actor_critic.load_state_dict(payload["model"], strict=True)
            trainer.optimizer.load_state_dict(payload["optimizer"])
            trainer.train_meter = cloudpickle.loads(payload["meter"])
            (self.out / "sampling.json").write_bytes(payload["sampling"])
            vector_restore(trainer, payload["vector"])
            random.setstate(payload["python_rng"])
            import numpy as np
            np.random.set_state(payload["numpy_rng"])
            torch.set_rng_state(payload["torch_rng"].cpu())
            torch.cuda.set_rng_state_all([s.cpu() for s in payload["cuda_rng"]])
            self.iterations, self.steps, self.updates = metadata["training_iterations"], metadata["training_env_steps"], metadata["optimizer_updates"]
            self.elapsed_before = metadata["wall_time_seconds"]
            obs = {key: value.to(trainer.device) for key, value in payload["obs"].items()}
        else:
            trainer.save_sampled_agent_seq(0)
            obs = trainer.envs.reset()
        # Server acceptance: replay 16 deterministic zero-action steps twice, then restore.
        # Probe transitions are excluded from training and fully rolled back.
        state = vector_capture(trainer)
        action = torch.zeros((32, trainer.actor_critic.num_actions), device=trainer.device)
        expected = [trainer.envs.step(action) for _ in range(16)]
        vector_restore(trainer, state)
        for result in expected:
            equality(trainer.envs.step(action), result)
        vector_restore(trainer, state)
        write_json(self.out / "status/resume_roundtrip.json", {"STATE_ROUNDTRIP": "PASS", "probe_lane_transitions": 512})
        self.started = time.monotonic()
        self.training_wall = self.elapsed_before
        return obs, self.iterations

    def finite(self, *values):
        import torch
        for value in values:
            if not torch.isfinite(value).all():
                self.nan_count += 1
                write_json(self.out / "status/numerical_failure.json", {"NaN_count": self.nan_count, "numerical_failure_count": self.numerical_failures})
                raise FloatingPointError("non-finite training tensor")

    def on_step(self, obs, reward, infos):
        self.finite(*obs.values(), reward)
        self.numerical_failures += sum(bool(info.get("mj_step_error", False)) for info in infos)
        if self.numerical_failures:
            write_json(self.out / "status/numerical_failure.json", {"NaN_count": self.nan_count, "numerical_failure_count": self.numerical_failures})
            raise FloatingPointError("native MuJoCo reported a numerical step failure")
        self.steps += 32

    def on_batch(self, val, logp, ent, kl):
        self.finite(val, logp, ent)
        import math
        if not math.isfinite(kl):
            self.nan_count += 1
            write_json(self.out / "status/numerical_failure.json", {"NaN_count": self.nan_count, "numerical_failure_count": self.numerical_failures})
            raise FloatingPointError("non-finite KL")
        self.entropy.append(float(ent.detach().cpu()))

    def after_optimizer_step(self, trainer):
        self.updates += 1

    def after_iteration(self, trainer, iteration, obs):
        import numpy as np
        self.finite(*trainer.actor_critic.parameters())
        self.iterations = iteration + 1
        assert self.steps == self.iterations * 81920, "transition accounting mismatch"
        wall = self.elapsed_before + time.monotonic() - self.started
        self.training_wall = wall
        meter = trainer.train_meter
        stats = {"iteration": self.iterations, "env_steps": self.steps, "effective_env_steps": self.steps,
                 "optimizer_updates": self.updates, "wall_time_seconds": wall, "throughput_env_steps_per_second": self.steps/wall,
                 "learning_rate": trainer.optimizer.param_groups[0]["lr"], "NaN_count": self.nan_count,
                 "numerical_failure_count": self.numerical_failures,
                 "entropy": float(np.mean(self.entropy)) if self.entropy else None}
        for label, key in (("policy_loss", "pi_loss"), ("value_loss", "val_loss"), ("KL", "approx_kl")):
            values = meter.train_stats[key]
            stats[label] = float(np.mean(values[-128:])) if values else None
        rewards = meter.mean_ep_rews["reward"]
        stats["mean_episodic_return"] = float(rewards[-1]) if rewards else None
        lengths = [float(v) for m in meter.agent_meters.values() for v in m.ep_len]
        stats["episode_length"] = float(np.mean(lengths)) if lengths else None
        self.entropy.clear()
        with (self.out / "logs/training.jsonl").open("a") as stream:
            stream.write(json.dumps(stats, allow_nan=False) + "\n")
        print(json.dumps(stats), flush=True)
        if self.iterations % 10 == 0 or self.iterations == accounting(self.budget)["training_iterations"]:
            self.save(trainer, obs, wall)

    def save(self, trainer, obs, wall):
        import cloudpickle
        import numpy as np
        import torch
        from metamorph.algos.ppo.envs import get_ob_rms
        completed = self.out / "checkpoints" / f"iteration_{self.iterations:06d}"
        directory = self.out / "checkpoints" / f"pending_{self.iterations:06d}_{os.getpid()}"
        directory.mkdir()  # interrupted pending directories remain separate from valid checkpoints
        inference = directory / "Unimal-v0.pt"
        torch.save([trainer.actor_critic, get_ob_rms(trainer.envs)], inference)
        torch.save({"model": trainer.actor_critic.state_dict(), "optimizer": trainer.optimizer.state_dict(),
                    "meter": cloudpickle.dumps(trainer.train_meter), "vector": vector_capture(trainer),
                    "sampling": (self.out / "sampling.json").read_bytes(), "obs": obs,
                    "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, directory / "resume.pt")
        metadata = {"signature": self.signature, "training_env_steps": self.steps, "training_iterations": self.iterations,
                    "optimizer_updates": self.updates, "seed": 1409, "train_morphology_count": 100,
                    "git_sha": self.signature["git_sha"], "config_sha256": self.signature["config_sha256"],
                    "checkpoint_sha256": sha256(inference), "reduced_budget": True, "formal_matched_baseline": False,
                    "wall_time_seconds": wall, "hashes": {name: sha256(directory/name) for name in ("Unimal-v0.pt", "resume.pt")}}
        write_json(directory / "metadata.json", metadata)
        directory.rename(completed)  # fails if a valid checkpoint already exists
        directory = completed
        pointer = {"path": str(directory.relative_to(self.out)), "metadata_sha256": sha256(directory / "metadata.json")}
        temporary = self.out / "status/last_valid.pending.json"
        write_json(temporary, pointer)
        temporary.replace(self.out / "status/last_valid.json")


def train(args):
    import torch
    from metamorph.utils import sample as su
    from metamorph.algos.ppo.ppo import PPO
    out = output_path(args.output)
    if args.resume:
        if not out.is_dir():
            raise FileNotFoundError("resume output does not exist")
    else:
        out.mkdir(parents=True, exist_ok=False)
        for folder in ("checkpoints", "logs", "configs", "provenance", "status", "scratch"):
            (out / folder).mkdir()
    cfg = configure(out, args.budget, args.workers, args.num_envs)
    split = validate_split()
    config_bytes = cfg.dump().encode()
    import hashlib
    import mujoco_py, numpy, gym, cloudpickle, lxml
    signature = {"seed": 1409, "config_sha256": hashlib.sha256(config_bytes).hexdigest(), "split": split,
                 "source_hashes": source_signature(), "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                 "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                 "torch": torch.__version__, "python": sys.version, "gpu_name": torch.cuda.get_device_name(0),
                 "mujoco_py": mujoco_py.__version__, "numpy": numpy.__version__, "gym": gym.__version__,
                 "cloudpickle": cloudpickle.__version__, "lxml": lxml.__version__}
    if args.mode == "pilot":
        benchmark_signature = json.loads(output_path(args.selection).read_text())["signature"]
        fields = ("torch", "python", "gpu_name", "mujoco_py", "numpy", "gym", "cloudpickle", "lxml")
        resume_match({key: benchmark_signature[key] for key in fields}, {key: signature[key] for key in fields})
    if args.resume:
        resume_match(json.loads((out / "provenance/signature.json").read_text()), signature)
    else:
        (out / "configs/resolved.yaml").write_bytes(config_bytes)
        write_json(out / "provenance/signature.json", signature)
    su.set_seed(1409)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.set_num_threads(1)  # existing native learner setting
    observer = PilotObserver(out, signature, args.budget, args.resume)
    trainer = PPO()
    resources = Resources(out)
    resources.future = resources.executor.submit(resources.collect)
    try:
        trainer.train(observer)
        telemetry = resources.finish()
        pointer = json.loads((out / "status/last_valid.json").read_text())
        final = out / pointer["path"] / "Unimal-v0.pt"
        # Fresh inference process validates original checkpoint/architecture/RMS compatibility.
        subprocess.run([sys.executable, str(ROOT / "tools/modumorph_inference_worker.py"),
                        "--checkpoint", str(final), "--config", str(out / "configs/resolved.yaml"),
                        "--device", "cpu"], input='', text=True, check=True, stdout=subprocess.DEVNULL)
        summary = {**accounting(args.budget), **telemetry, "NUM_ENVS": 32, "CPU_WORKERS": args.workers,
                   "env_steps_per_second": observer.steps/observer.training_wall,
                   "NaN_count": observer.nan_count, "numerical_failure_count": observer.numerical_failures,
                   "STATE_ROUNDTRIP": "PASS", "CHECKPOINT_ADAPTER_LOAD": "PASS", "status": "PASS",
                   "reduced_budget": True, "formal_matched_baseline": False, "checkpoint": str(final), "signature": signature,
                   "training_state_digest": training_digest(trainer, observer)}
        write_json(out / "status/summary.json", summary)
        print("LEARNED_CHECKPOINT=" + str(final))
        print("NEXT_STEP=1-walker real server smoke; no evaluation launched")
    finally:
        if not resources.stop.is_set():
            resources.stop.set()
        resources.executor.shutdown()
        trainer.writer.close()
        trainer.envs.close()


def benchmark(args, hardware):
    out = output_path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "hardware.json", hardware)
    trials = [16] + [n for n in hardware["candidates"] if n != 16]
    results = []
    for workers in trials:
        trial = out / f"workers_{workers}"
        print(f"[BENCHMARK] NUM_ENVS=32 CPU_WORKERS={workers}", flush=True)
        with (out / f"workers_{workers}.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "internal",
                                        "--workers", str(workers), "--budget", "245760", "--output", str(trial)],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            # Hard deadline covers initialization, rollout, learner, snapshot and checkpoint load.
            timed_out = threading.Event()
            def expire():
                import signal
                timed_out.set()
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            # New session below avoids signalling the benchmark coordinator.
            timer = threading.Timer(900, expire)
            timer.start()
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                rc = process.wait()
            finally:
                timer.cancel()
        summary = trial / "status/summary.json"
        if rc == 0 and not timed_out.is_set() and summary.is_file():
            result = json.loads(summary.read_text())
            results.append(result)
        else:
            write_json(out / f"workers_{workers}_failure.json", {"exit_code": rc, "timed_out": timed_out.is_set(), "status": "FAIL"})
        write_json(out / "results.json", results)
    if not results or not any(r["CPU_WORKERS"] == 16 for r in results):
        raise RuntimeError("native default benchmark must pass before selecting acceleration")
    native = next(r for r in results if r["CPU_WORKERS"] == 16)
    equivalent = [r for r in results if r["training_state_digest"] == native["training_state_digest"]]
    for result in results:
        result["DEFAULT_TRAJECTORY_EQUIVALENCE"] = "PASS" if result in equivalent else "FAIL"
    write_json(out / "results.json", results)
    selected = max(equivalent, key=lambda r: r["env_steps_per_second"])
    write_json(out / "selected.json", selected)
    print("SELECTED_CPU_WORKERS=" + str(selected["CPU_WORKERS"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "benchmark", "pilot", "internal"))
    parser.add_argument("--output")
    parser.add_argument("--budget", type=int, default=10_000_000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--selection", default="./tmp/modumorph_throughput_s1409/selected.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    hardware = preflight()
    if args.mode == "preflight":
        return
    if args.mode == "benchmark":
        args.output = args.output or "./tmp/modumorph_throughput_s1409"
        benchmark(args, hardware)
    elif args.mode == "pilot":
        if args.budget != 10_000_000:
            raise ValueError("seed-1409 pilot budget is fixed to 10M transitions")
        args.output = args.output or "./tmp/modumorph_pilot_s1409_10m"
        selected = json.loads(output_path(args.selection).read_text())
        if selected["status"] != "PASS" or selected["NaN_count"] or selected["numerical_failure_count"]:
            raise ValueError("selected benchmark is not stable")
        if selected["signature"]["source_hashes"] != source_signature():
            raise ValueError("benchmark source has changed; rerun benchmark")
        if selected["signature"]["split"] != validate_split():
            raise ValueError("training assets changed since benchmark")
        if selected["DEFAULT_TRAJECTORY_EQUIVALENCE"] != "PASS":
            raise ValueError("benchmark did not preserve the default trajectory")
        for key, value in selected["signature"]["thread_environment"].items():
            if os.environ.get(key) != value:
                raise ValueError("thread environment changed since benchmark: " + key)
        args.workers = selected["CPU_WORKERS"]
        train(args)
    else:
        if args.budget != 245760 or args.resume:
            raise ValueError("internal mode is limited to a fresh 245760-transition benchmark trial")
        train(args)


if __name__ == "__main__":
    main()
