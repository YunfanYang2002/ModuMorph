"""Opt-in nominal UNIMAL state snapshots, gated by real server replay equality."""
import random
import copy
import json
import tempfile
from pathlib import Path

import cloudpickle
import numpy as np

FIELDS = ("qacc_warmstart", "ctrl", "qfrc_applied", "xfrc_applied",
          "mocap_pos", "mocap_quat", "userdata", "qacc", "qfrc_actuator",
          "body_xpos", "body_xquat", "body_xmat", "geom_xpos", "geom_xmat",
          "site_xpos", "site_xmat", "sensordata", "cvel")

FIELD_DIMENSIONS = {
    "qacc_warmstart": "nv", "ctrl": "nu", "qfrc_applied": "nv",
    "xfrc_applied": "nbody", "mocap_pos": "nmocap", "mocap_quat": "nmocap",
    "userdata": "nuserdata", "qacc": "nv", "qfrc_actuator": "nv",
    "body_xpos": "nbody", "body_xquat": "nbody", "body_xmat": "nbody",
    "geom_xpos": "ngeom", "geom_xmat": "ngeom", "site_xpos": "nsite",
    "site_xmat": "nsite", "sensordata": "nsensordata", "cvel": "nbody",
}
_reported_none_fields = set()


def capture_integration(sim):
    integration = {}
    none_fields = []
    for key in FIELDS:
        value = getattr(sim.data, key)
        if value is None:
            dimension = FIELD_DIMENSIONS[key]
            size = getattr(sim.model, dimension)
            if size != 0:
                raise ValueError(f"snapshot field {key} is unexpectedly None: model.{dimension}={size}")
            none_fields.append(key)
            integration[key] = None
        elif np.isscalar(value):
            integration[key] = value
        else:
            integration[key] = copy.deepcopy(value)
    new_none_fields = sorted(set(none_fields) - _reported_none_fields)
    if new_none_fields:
        print("SNAPSHOT_NONE_FIELDS=" + json.dumps(new_none_fields), flush=True)
        _reported_none_fields.update(new_none_fields)
    return integration


def restore_integration(sim, integration):
    if integration.keys() != set(FIELDS):
        raise ValueError("snapshot integration field set mismatch")
    for key, saved in integration.items():
        runtime = getattr(sim.data, key)
        if saved is None or runtime is None:
            if saved is not None or runtime is not None:
                raise ValueError(f"snapshot field {key} None/value representation mismatch")
            dimension = FIELD_DIMENSIONS[key]
            size = getattr(sim.model, dimension)
            if size != 0:
                raise ValueError(f"snapshot field {key} is unexpectedly None: model.{dimension}={size}")
            continue
        if type(saved) is not type(runtime):
            raise TypeError(f"snapshot field {key} representation mismatch: {type(saved).__name__} != {type(runtime).__name__}")
        saved_array, runtime_array = np.asarray(saved), np.asarray(runtime)
        if saved_array.shape != runtime_array.shape:
            raise ValueError(f"snapshot field {key} shape mismatch: {saved_array.shape} != {runtime_array.shape}")
        if saved_array.dtype != runtime_array.dtype:
            raise TypeError(f"snapshot field {key} dtype mismatch: {saved_array.dtype} != {runtime_array.dtype}")
        if np.isscalar(runtime) or isinstance(runtime, tuple):
            setattr(sim.data, key, copy.deepcopy(saved))
        elif isinstance(runtime, np.ndarray):
            runtime[...] = saved
        else:
            runtime[:] = copy.deepcopy(saved)


def capture(env):
    layers = []
    current = env
    while True:
        fields = current.__dict__.copy()
        link = "_env" if "_env" in fields else "env" if "env" in fields else None
        if link is None:
            sim = fields.pop("sim")
            if fields["viewer"] is not None or fields["_viewers"]:
                raise ValueError("training snapshot forbids active render viewers")
            layers.append((type(current), fields, None))
            break
        child = fields.pop(link)
        layers.append((type(current), fields, link))
        current = child
    mjb = sim.model.get_mjb()
    if not isinstance(mjb, bytes) or not mjb:
        raise ValueError("snapshot model MJB must be non-empty bytes")
    return {"layers": cloudpickle.dumps(layers), "mjb": mjb,
            "sim_state": sim.get_state(),
            "integration": capture_integration(sim)}


def restore(snapshot, scratch):
    import mujoco_py
    from tools.modumorph_training_contract import output_path
    scratch = output_path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=scratch, suffix=".mjb", delete=False) as stream:
        stream.write(snapshot["mjb"])
        model_path = Path(stream.name)
    try:
        model_bytes = model_path.read_bytes()
        if not model_bytes:
            raise ValueError("snapshot MJB file is empty: " + str(model_path))
        sim = mujoco_py.MjSim(mujoco_py.load_model_from_mjb(model_bytes))
    finally:
        if model_path.exists():
            model_path.unlink()
    sim.set_state(snapshot["sim_state"])
    sim.forward()
    restore_integration(sim, snapshot["integration"])
    child = None
    for cls, fields, link in reversed(cloudpickle.loads(snapshot["layers"])):
        instance = object.__new__(cls)
        instance.__dict__.update(fields)
        if link is None:
            instance.sim = sim
        else:
            instance.__dict__[link] = child
        child = instance
    return child


def capture_worker(envs):
    return {"envs": [capture(env) for env in envs],
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}


def restore_worker(state, scratch):
    envs = [restore(env, scratch) for env in state["envs"]]
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    return envs
