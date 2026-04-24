# GR00T + Isaac Lab + G1 + Ray Serve Demo

End-to-end infrastructure for serving NVIDIA's GR00T N1.7 VLA model to
Isaac Lab Unitree G1 simulations across an Anyscale Ray cluster.

## Status

**WIP — last updated 2026-04-24.** Components verified working independently;
end-to-end rollout pipeline is the remaining bug.

| Component | Status | Notes |
|---|---|---|
| Isaac Lab + G1 + pick-place scene | ✅ Working | After NVIDIA pinocchio workaround |
| GR00T N1.7-3B inference | ✅ Working | ~720ms per call, 3B params on 1 A10G |
| Ray Serve HTTP policy deployment | ✅ Working | Both placeholder and GR00T deploy |
| Sim worker subprocess | ✅ Boots Isaac Sim, queries policy | Hangs after first few steps |
| End-to-end rollout + GIF | 🔴 In progress | Subprocess hang during step loop |

## Architecture

```
head node (CPU)
│
├── Ray Serve cluster (lives on a GPU worker)
│     └── GR00TPolicyServer  [POST /predict] [GET /stats]
│           Loads nvidia/GR00T-N1.7-3B with REAL_G1 embodiment
│
└── N sim worker tasks (one per GPU worker)
      Each Ray task shell-execs:
          python sim_worker.py --policy-url http://HEAD:8000 ...
      Sim queries the policy via HTTP, executes actions, renders frames.
```

**Why subprocess instead of Ray actors:** Isaac Sim uses `omni.kit.async_engine`
which expects a main asyncio event loop. Inside Ray actor threads,
`MainEventLoopWrapper.g_main_event_loop` is None → scene loading crashes with
`AttributeError: 'NoneType' object has no attribute 'create_task'`. Subprocess
gives Isaac Sim a clean Python interpreter + fresh event loop.

## Files

- `g1_env.py` — Isaac Lab G1 env wrapper. Translates obs → GR00T nested dict
  (video/state/language) and GR00T action chunks → Isaac Lab's `(1, 28)`
  flat action tensor. **Includes the NVIDIA pinocchio #4090 workaround**
  (import `pinocchio` BEFORE `AppLauncher`).
- `policy_server.py` — Ray Serve deployment of `Gr00tPolicy`. Wraps with
  `@serve.ingress(FastAPI())` to expose HTTP `POST /predict`. Applies two
  runtime patches before model load: `VideoInput` shim (moved to
  `transformers.video_utils` in 4.54+) and `flash_attention_2` force on
  Qwen3 VLM backbone.
- `sim_worker.py` — Standalone script (NOT a Ray actor). Loads `g1_env`,
  loops `env.step` querying policy via HTTP, saves a GIF. Run as a
  subprocess from `run_demo.py`.
- `run_demo.py` — Orchestrator. Starts Ray Serve, deploys policy, spawns
  N sim worker subprocesses on Ray tasks, collects results.
- `setup_workers.sh` — Initial worker bringup (Isaac-GR00T install,
  flash-attn prebuilt wheel, conda env patches).

## Verified runtime patches

These are required because we're running GR00T against transformers 4.57.6
instead of the pinned 4.51.3:

1. **VideoInput shim**: In transformers ≥4.54, `VideoInput` was moved from
   `transformers.image_utils` to `transformers.video_utils`. Eagle's
   dynamic processor still imports from the old location.
   ```python
   import transformers.image_utils
   from transformers.video_utils import VideoInput
   transformers.image_utils.VideoInput = VideoInput
   ```

2. **flash_attention_2 force**: Qwen3 VLM asserts
   `_attn_implementation == "flash_attention_2"` but `AutoModel.from_config`
   doesn't propagate `attn_implementation` kwarg through. Monkey-patch
   `_BaseAutoModelClass.from_config`.

3. **HF_TOKEN propagation**: Cosmos-Reason2-2B (used as N1.7's VLM backbone)
   is gated. Worker subprocesses need `HF_TOKEN` in env; we propagate via
   Ray's `runtime_env={"env_vars": {"HF_TOKEN": ...}}`.

4. **Pinocchio pre-import**: NVIDIA IsaacLab issue #4090. Pinocchio's C++
   `std::vector<std::string>` binding gets corrupted after Isaac Lab loads
   a robot URDF. Workaround: `import pinocchio` before `AppLauncher`.
   Confirmed by NVIDIA on the issue tracker.

## Embodiment + obs/action schema (verified)

Embodiment tag: `EmbodimentTag.REAL_G1` (NOT `UNITREE_G1` — that's a
posttrain-only tag requiring a fine-tuned checkpoint).

**Obs format** (nested dict):
```python
{
    "video":    {"ego_view":  np.ndarray (B, 2, H, W, 3) uint8},
    "state": {
        "left_wrist_eef_9d":  (B, 1, 9) float32,  # 3 pos + 6 rot (R[0:2].flat)
        "right_wrist_eef_9d": (B, 1, 9) float32,
        "left_hand":          (B, 1, 7) float32,
        "right_hand":         (B, 1, 7) float32,
        "left_arm":           (B, 1, 7) float32,
        "right_arm":          (B, 1, 7) float32,
        "waist":              (B, 1, 3) float32,
    },
    "language": {"annotation.human.task_description": [["pick up the apple"]]},
}
```

**Action chunk** (40-step horizon):
```python
{
    "left_wrist_eef_9d":     (1, 40, 9),
    "right_wrist_eef_9d":    (1, 40, 9),
    "left_hand":             (1, 40, 7),
    "right_hand":            (1, 40, 7),
    "left_arm":              (1, 40, 7),
    "right_arm":             (1, 40, 7),
    "waist":                 (1, 40, 3),
    "base_height_command":   (1, 40, 1),
    "navigate_command":      (1, 40, 3),
}
```

Isaac Lab task `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0` expects
action shape `(1, 28)`. Current `_flatten_action` packs
`left_arm + right_arm + left_hand + right_hand` (7+7+7+7=28). May need
remapping based on actual joint ordering.

## Setup

```bash
# On head node
bash setup_workers.sh

# Login to HF (token stored at ~/.cache/huggingface/token)
hf auth login
# Make sure to accept https://huggingface.co/nvidia/Cosmos-Reason2-2B

# Test pieces individually
python test_g1_sim.py            # verify Isaac Lab + G1 scene loads
python test_groot_standalone.py  # verify GR00T N1.7-3B inference
```

## Run

```bash
# Architecture test (random actions, no GR00T weights)
python run_demo.py --placeholder --num-workers 1 --episodes 1 --max-steps 30

# Full GR00T demo
python run_demo.py --num-workers 2 --episodes 1 --max-steps 200
```

## Known issues

- **Sim worker hangs after first few env.step calls** in placeholder mode
  (last verified 2026-04-24 ~22:00). Action format passes type checks
  (torch tensor on correct device), but the subprocess stops producing
  output. Not yet diagnosed. Likely candidates:
  - `env.render()` fails silently in fully-headless mode → no frames →
    something in `_format_obs` blocks waiting for next obs;
  - `_format_obs` accesses an obs key that's not present (e.g.
    `policy_obs["left_eef_pos"]`) and the subprocess crashes before
    flushing logs;
  - 28-dim action vector is the wrong layout for this task and physics
    diverges into NaN, but error gets swallowed.
- **Action translation is unverified**: GR00T outputs 9 action keys for
  REAL_G1; Isaac Lab task wants `(1, 28)`. Current packing is
  `left_arm[:7] + right_arm[:7] + left_hand[:7] + right_hand[:7]`. The
  `wrist_eef_9d` keys are dropped. This is almost certainly wrong for
  this specific task cfg.
- **Camera obs is duplicated, not buffered**: GR00T's video horizon is
  `[-20, 0]` (2 frames). We currently duplicate the current frame for
  both slots since we don't have a 20-step lookback ring buffer. Real
  policy would benefit from a real history buffer.

## Acknowledgments

- NVIDIA team for the [pinocchio #4090 workaround](https://github.com/isaac-sim/IsaacLab/issues/4090)
- Isaac-GR00T `n1.6-release` and `main` branches for embodiment configs
- Ian for the Isaac Lab template baseline pattern
