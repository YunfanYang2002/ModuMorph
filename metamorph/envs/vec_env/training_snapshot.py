"""Opt-in nominal UNIMAL state snapshots, gated by real server replay equality."""
import random
import tempfile
from pathlib import Path

import cloudpickle
import numpy as np

FIELDS = ("qacc_warmstart", "ctrl", "qfrc_applied", "xfrc_applied",
          "mocap_pos", "mocap_quat", "userdata", "qacc", "qfrc_actuator",
          "body_xpos", "body_xquat", "body_xmat", "geom_xpos", "geom_xmat",
          "site_xpos", "site_xmat", "sensordata", "cvel")


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
    return {"layers": cloudpickle.dumps(layers), "mjb": sim.model.get_mjb(),
            "sim_state": sim.get_state(),
            "integration": {key: getattr(sim.data, key).copy() for key in FIELDS}}


def restore(snapshot, scratch):
    import mujoco_py
    from tools.modumorph_training_contract import output_path
    scratch = output_path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=scratch, suffix=".mjb", delete=False) as stream:
        stream.write(snapshot["mjb"])
        model_path = Path(stream.name)
    try:
        sim = mujoco_py.MjSim(mujoco_py.load_model_from_mjb(str(model_path)))
    finally:
        model_path.unlink()
    sim.set_state(snapshot["sim_state"])
    sim.forward()
    for key, value in snapshot["integration"].items():
        getattr(sim.data, key)[:] = value
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
