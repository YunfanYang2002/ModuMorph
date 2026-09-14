"""Original ModuMorph namespace only; stdout is a strict JSON-lines protocol."""
from __future__ import annotations
import argparse
import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modumorph_policy_adapter import PROPRIO_TYPES, CONTEXT_TYPES


def load_policy(checkpoint_path, config_path, device):
    import numpy as np
    import torch
    from metamorph.config import cfg
    from metamorph.algos.ppo.model import ActorCritic
    cfg.merge_from_file(str(config_path))
    if cfg.MODEL.TYPE != "transformer" or not cfg.MODEL.TRANSFORMER.FIX_ATTENTION or not cfg.MODEL.TRANSFORMER.HYPERNET:
        raise ValueError("ModuMorph baseline requires the original fixed-attention hypernetwork policy")
    if cfg.ENV_NAME != "Unimal-v0" or cfg.MODEL.PROPRIOCEPTIVE_OBS_TYPES != PROPRIO_TYPES or cfg.MODEL.CONTEXT_OBS_TYPES != CONTEXT_TYPES:
        raise ValueError("checkpoint config is not the audited UNIMAL observation contract")
    if cfg.MODEL.OBS_TO_NORM != ["proprioceptive"] or cfg.ENV.KEYS_TO_KEEP:
        raise ValueError("unsupported normalization or external observation keys")
    if cfg.MIRROR_DATA_AUG or cfg.MODEL.TRANSFORMER.USE_SWAT_PE or cfg.MODEL.TRANSFORMER.USE_SWAT_RE:
        raise ValueError("mirror/SWAT requires a separately proven ordering binding")
    if cfg.MODEL.TRANSFORMER.PER_NODE_EMBED or cfg.MODEL.TRANSFORMER.PER_NODE_DECODER:
        raise ValueError("per-training-ID weights do not provide held-out morphology inference")
    cfg.PPO.NUM_ENVS = 1
    cfg.DEVICE = device
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if not isinstance(payload, (list, tuple)) or len(payload) != 2 or not isinstance(payload[0], ActorCritic):
        raise TypeError("expected original [ActorCritic, ob_rms] checkpoint")
    model, rms = payload
    if not isinstance(rms, dict) or set(rms) != {"proprioceptive"}:
        raise ValueError("checkpoint must contain proprioceptive normalization statistics")
    if model.seq_len != cfg.MODEL.MAX_LIMBS or model.num_actions != cfg.MODEL.MAX_LIMBS * 2:
        raise ValueError("checkpoint/config limb or action dimensions disagree")
    if tuple(rms["proprioceptive"].mean.shape) != (cfg.MODEL.MAX_LIMBS * 52,):
        raise ValueError("checkpoint normalization shape is not L*52")
    stats = rms["proprioceptive"]
    if stats.var.shape != stats.mean.shape or not np.isfinite(stats.mean).all() or not np.isfinite(stats.var).all() or np.any(stats.var < 0) or not np.isfinite(stats.count) or stats.count <= 0:
        raise ValueError("invalid checkpoint observation normalization statistics")
    expected_args = cfg.MODEL.TRANSFORMER if cfg.MODEL.TYPE == "transformer" else cfg.MODEL.MLP
    for network in (model.mu_net, model.v_net):
        if network.model_args != expected_args:
            raise ValueError("serialized architecture settings differ from supplied config")
    spaces = {"proprioceptive": SimpleNamespace(shape=(cfg.MODEL.MAX_LIMBS * 52,)),
              "context": SimpleNamespace(shape=(cfg.MODEL.MAX_LIMBS * 35,))}
    expected = ActorCritic(spaces, None)
    signature = lambda net: {key: tuple(value.shape) for key, value in net.state_dict().items()}
    if signature(model) != signature(expected):
        raise ValueError("checkpoint parameter names/shapes differ from configured architecture")
    del expected
    model.to(device).eval().requires_grad_(False)
    return model, rms, cfg


def original_context(snapshot, data_snapshot, padding, action_padding, cfg):
    import numpy as np
    from metamorph.envs.modules.agent import Agent
    agent = Agent()
    model = SimpleNamespace(**{key: np.asarray(value) for key, value in snapshot.items()})
    agent.agent_body_idxs = np.arange(model.body_pos.shape[0])
    agent.agent_geom_idxs = np.arange(model.geom_quat.shape[0])
    data = SimpleNamespace(**{key: np.asarray(value) for key, value in data_snapshot.items()})
    def body_position(name):
        if name != "torso/0":
            raise ValueError(f"unexpected body lookup: {name}")
        return data.body_xpos[0]
    data.get_body_xpos = body_position
    sim = SimpleNamespace(model=model, data=data)
    limb, joint = agent.get_context(sim)
    valid_limbs = int((~np.asarray(padding, dtype=bool)).sum())
    slots = np.asarray(action_padding, dtype=bool)[:valid_limbs * 2]
    padded = np.zeros((valid_limbs * 2, joint.shape[1]))
    padded[~slots] = joint
    rows = np.hstack((limb, padded.reshape(valid_limbs, -1)))
    rows = np.pad(rows, ((0, cfg.MODEL.MAX_LIMBS - valid_limbs), (0, 0)))
    agent.joint_mask_for_node_graph = (~slots).tolist()
    native = agent.combine_limb_joint_obs(agent.get_limb_obs(sim), agent.get_joint_obs(sim),
                                         SimpleNamespace(metadata={"mirrored": False}))
    native = np.pad(native, ((0, cfg.MODEL.MAX_LIMBS - valid_limbs), (0, 0)))
    return {"context": rows.ravel().tolist(), "proprioceptive": native.ravel().tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    # Model initialization/unpickling can print; keep protocol stdout clean.
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        model, rms, cfg = load_policy(Path(args.checkpoint), Path(args.config), args.device)
    stats = rms["proprioceptive"]
    print(json.dumps({"max_limbs": cfg.MODEL.MAX_LIMBS, "architecture": type(model).__name__,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "config_seed": int(cfg.RNG_SEED), "training_seed": "UNKNOWN", "training_steps": "UNKNOWN",
                      "rms": {"mean": stats.mean.tolist(), "var": stats.var.tolist(), "count": float(stats.count)}}, allow_nan=False), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            if request["op"] == "context":
                response = original_context(request["model"], request["data"], request["mask"], request["act_mask"], cfg)
            elif request["op"] == "act":
                obs = {key: torch.tensor(value, dtype=torch.float32, device=args.device) for key, value in request["obs"].items()}
                _, distribution, _, _, _, _ = model(obs, compute_val=False)
                action = distribution.mean
                if not torch.isfinite(action).all():
                    raise ValueError("non-finite model output")
                response = {"action": action.cpu().tolist()}
            else:
                raise ValueError(f"unknown worker operation: {request['op']}")
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
