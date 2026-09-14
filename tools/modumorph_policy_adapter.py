"""Static ModuMorph inference bridge; contains no evaluation/physics logic."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROPRIO_TYPES = ["body_xpos", "body_xvelp", "body_xvelr", "body_xquat",
                "body_pos", "body_ipos", "body_iquat", "geom_quat",
                "body_mass", "body_shape", "qpos", "qvel", "jnt_pos",
                "joint_range", "joint_axis", "gear"]
CONTEXT_TYPES = ["body_pos", "body_ipos", "body_iquat", "geom_quat",
                 "body_mass", "body_shape", "jnt_pos", "joint_range", "joint_axis", "gear"]
# Original Agent.get_context bounds, in the original configured field order.
CONTEXT_FIELDS = [
    (13, 16, [-.5, -.45, -.49], [.5, .45, 2.04]),
    (16, 19, [-.225, -.225, -.225], [.225, .225, 0.]),
    (19, 23, [.70710678, -.70710678, -.70710678, 0.], [1., .70710678, .70710678, 0.]),
    (23, 27, [.70710678, -.70710678, -.70710678, 0.], [1., .70710678, .70710678, 0.]),
    (27, 28, [1.17809725], [4.1887902]),
    (28, 30, [.05, 0.], [.1, .22627417]),
]
JOINT_FIELDS = [(2, 5, [-.05, -.05, 0.], [.05, .05, .05]),
                (5, 7, [-1.57079633, 0.], [0., 1.57079633]),
                (7, 10, [-.5000024, -.5000024, -1.], [1., 1., 1.]),
                (10, 11, [0.], [300.])]


def static_context(raw_proprio, padding, act_padding):
    """Convert *unfiltered nominal* node features, never invert clipped obs."""
    padding = np.asarray(padding)
    act_padding = np.asarray(act_padding)
    raw = np.asarray(raw_proprio)
    if padding.ndim != 1 or raw.shape != (padding.size * 52,) or act_padding.shape != (padding.size * 2,):
        raise ValueError("static context requires L*52 raw features, L mask and 2L action mask")
    if not np.isin(padding, [0, 1]).all() or not np.isin(act_padding, [0, 1]).all():
        raise ValueError("padding masks must be binary")
    valid_limbs = int((padding == 0).sum())
    if valid_limbs == 0 or not np.array_equal(padding, np.arange(padding.size) >= valid_limbs):
        raise ValueError("real limbs must form a nonempty prefix")
    if not act_padding[:2].all() or not act_padding[valid_limbs * 2:].all():
        raise ValueError("torso and padded limbs cannot contain actuators")
    if not np.isfinite(raw).all():
        raise ValueError("non-finite raw morphology observation")
    rows = raw.reshape(padding.size, 52)
    def transform(values, fields):
        result = []
        for start, end, lower, upper in fields:
            low, high = np.asarray(lower), np.asarray(upper)
            result.append(-1. * (low != high) + 2. * (values[:, start:end] - low) / (high - low + 1e-8))
        return np.concatenate(result, axis=1)
    limb = transform(rows, CONTEXT_FIELDS)
    joints = rows[:, 30:].reshape(-1, 11)
    joint = transform(joints, JOINT_FIELDS)
    # Original get_context transforms valid joints before scatter into zero slots.
    joint[act_padding.astype(bool)] = 0.
    context = np.concatenate([limb, joint.reshape(padding.size, 18)], axis=1)
    context[padding.astype(bool)] = 0.
    return context.ravel()


def validate_policy_observation(obs, max_limbs):
    shapes = {"proprioceptive": (1, max_limbs * 52), "obs_padding_mask": (1, max_limbs),
              "act_padding_mask": (1, max_limbs * 2)}
    for key, shape in shapes.items():
        value = obs[key]
        if tuple(value.shape) != shape or not np.isfinite(value.detach().cpu().numpy()).all():
            raise ValueError(f"{key}: expected finite {shape}, got {tuple(value.shape)}")
    for key in ("obs_padding_mask", "act_padding_mask"):
        if not np.isin(obs[key].detach().cpu().numpy(), [0, 1]).all():
            raise ValueError(f"{key} must be binary")


class ModuMorphPolicyAdapter:
    """Separate interpreter prevents the two `metamorph` packages colliding."""
    def __init__(self, root, checkpoint, config, device, log):
        self.log = Path(log).open("x", encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(Path(root) / "tools/modumorph_inference_worker.py"),
             "--checkpoint", str(checkpoint), "--config", str(config), "--device", str(device)],
            cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding="utf-8")
        self.device = device
        self.bindings = {}
        self.metadata = self._receive()
        self.max_limbs = self.metadata["max_limbs"]
        self.walker = None

    def _receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"ModuMorph worker exited; inspect {self.log.name}")
        return json.loads(line)

    def _request(self, value):
        self.process.stdin.write(json.dumps(value, allow_nan=False) + "\n")
        self.process.stdin.flush()
        return self._receive()

    def bind(self, walker, raw, mask, act_mask, model_snapshot, data_snapshot):
        context = static_context(raw, mask, act_mask)
        original = self._request({"op": "context", "model": model_snapshot,
                                  "data": data_snapshot,
                                  "mask": np.asarray(mask).tolist(), "act_mask": np.asarray(act_mask).tolist()})
        expected = np.asarray(original["context"])
        # Both paths operate on the same double-precision nominal model values.
        np.testing.assert_array_equal(context, expected)
        np.testing.assert_array_equal(np.asarray(raw), np.asarray(original["proprioceptive"]))
        self.bindings[walker] = context.astype(np.float32)
        return context

    def reset(self, walker):
        if walker not in self.bindings:
            raise ValueError(f"walker has no exact static binding: {walker}")
        self.walker = walker

    def act(self, obs):
        import torch
        if self.walker is None:
            raise RuntimeError("adapter.reset(walker) required")
        validate_policy_observation(obs, self.max_limbs)
        # Allowlist: no history, previous action, privileged context or dynamics.
        inputs = {key: obs[key].detach().cpu().tolist() for key in
                  ("proprioceptive", "obs_padding_mask", "act_padding_mask", "edges")}
        inputs["context"] = self.bindings[self.walker][None, :].tolist()
        action = np.asarray(self._request({"op": "act", "obs": inputs})["action"], dtype=np.float32)
        if action.shape != (1, self.max_limbs * 2) or not np.isfinite(action).all():
            raise ValueError("ModuMorph mean action must be finite [1,2L]")
        # Unchanged mean command. Frozen environment removes padding and clips.
        return torch.from_numpy(action).to(self.device)

    def __call__(self, obs):
        return None, SimpleNamespace(mean=self.act(obs)), None, None

    def get_metadata(self):
        return self.metadata

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            rc = self.process.wait(timeout=30)
        else:
            rc = self.process.returncode
        self.log.close()
        if rc != 0:
            raise RuntimeError(f"ModuMorph worker failed with exit {rc}; see {self.log.name}")
