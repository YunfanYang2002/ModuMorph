"""Frozen seed-1409 pilot contract; no simulator imports."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = "/home/yyf/Workspace/Code/rmamorph/output/unimals_100/train"
RECIPE = ["PPO.KL_TARGET_COEF", "5.", "MODEL.TRANSFORMER.POS_EMBEDDING", "None",
          "MODEL.TRANSFORMER.EMBEDDING_DROPOUT", "False",
          "MODEL.TRANSFORMER.FIX_ATTENTION", "True", "MODEL.TRANSFORMER.HYPERNET", "True",
          "MODEL.TRANSFORMER.CONTEXT_ENCODER", "linear"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def output_path(path):
    resolved = Path(path).resolve()
    if (ROOT / "tmp").resolve() not in resolved.parents:
        raise ValueError("output must be a child of project ./tmp: " + str(resolved))
    return resolved


def accounting(budget, num_envs=32, horizon=2560):
    if budget < num_envs * horizon:
        raise ValueError("budget must cover one complete native rollout")
    iterations = budget // (num_envs * horizon)
    return {"requested_env_steps": budget, "training_iterations": iterations,
            "transitions_per_rollout": num_envs * horizon,
            "training_env_steps": iterations * num_envs * horizon,
            "unused_budget": budget % (num_envs * horizon)}


def configure(out, budget, workers, num_envs=32):
    from metamorph.config import cfg
    if num_envs != 32:
        raise ValueError("NUM_ENVS must remain 32: changing it changes PPO/sampler semantics")
    if workers not in (1, 2, 4, 8, 16, 32):
        raise ValueError("workers must divide the native 32 env lanes")
    cfg.merge_from_file(str(ROOT / "configs/ft.yaml"))
    cfg.merge_from_list(RECIPE)
    cfg.RNG_SEED = 1409
    cfg.ENV.WALKER_DIR = TRAIN_ROOT
    cfg.ENV.WALKERS = json.loads((ROOT / "configs/modumorph_training_binding.json").read_text())["walkers"]
    cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_JOINTS = 12, 16
    cfg.OUT_DIR = str(output_path(out))
    cfg.VECENV.IN_SERIES = 32 // workers
    cfg.PPO.MAX_ITERS = int(cfg.PPO.MAX_STATE_ACTION_PAIRS) // (32 * cfg.PPO.TIMESTEPS)
    cfg.PPO.EARLY_EXIT = True
    cfg.PPO.EARLY_EXIT_STATE_ACTION_PAIRS = float(budget)
    cfg.PPO.EARLY_EXIT_MAX_ITERS = accounting(budget)["training_iterations"]
    return cfg


def validate_split(root=TRAIN_ROOT):
    binding = json.loads((ROOT / "configs/modumorph_training_binding.json").read_text())
    root = Path(root).resolve()
    reference = root.parents[2] / binding["authority_config"]
    if text_hash(reference) != binding["authority_config_sha256"]:
        raise ValueError("frozen training authority config hash mismatch")
    import yaml
    if yaml.safe_load(reference.read_text())["ENV"]["WALKERS"] != binding["walkers"]:
        raise ValueError("frozen training walker order mismatch")
    expected = set(binding["walkers"])
    for folder, suffix in (("xml", ".xml"), ("metadata", ".json")):
        actual = {p.stem for p in (root / folder).iterdir() if p.is_file()}
        if actual != expected:
            raise ValueError(f"{folder} IDs mismatch: missing={expected-actual}, extra={actual-expected}")
    files = {}
    for name in binding["walkers"]:
        xml, metadata = root / "xml" / (name + ".xml"), root / "metadata" / (name + ".json")
        tree = ET.parse(xml)
        if tree.find(".//body[@name='torso/0']") is None or tree.find("actuator") is None or tree.find("sensor") is None:
            raise ValueError("native UNIMAL XML contract mismatch: " + name)
        data = json.loads(metadata.read_text())
        if not isinstance(data["dof"], int) or not isinstance(data["num_limbs"], int):
            raise ValueError("native metadata dof/num_limbs must be integer: " + name)
        if data["dof"] >= 16 or data["num_limbs"] + 1 >= 12:
            raise ValueError("walker exceeds native padding: " + name)
        files[name] = {"xml_sha256": sha256(xml), "metadata_sha256": sha256(metadata)}
    return {"root": str(root), "walkers": binding["walkers"], "files": files,
            "authority_config_sha256": binding["authority_config_sha256"]}


def resume_match(saved, current):
    if saved != current:
        changed = [k for k in set(saved) | set(current) if saved.get(k) != current.get(k)]
        raise ValueError("resume provenance mismatch: " + ", ".join(sorted(changed)))
