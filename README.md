# GR00T + Isaac Lab + G1 + Ray Serve

End-to-end infrastructure for serving NVIDIA GR00T VLA models to Isaac Lab Unitree G1 simulations on an Anyscale Ray cluster.

## What this demo shows

NVIDIA's GR00T humanoid foundation model controlling a Unitree G1 in NVIDIA Isaac Lab, with the model served via Ray Serve on Anyscale. Two parallel paths:

1. **Path A (Ray Serve HTTP)**: GR00T-N1.7-3B base model, REAL_G1 embodiment, served via Ray Serve with FastAPI ingress. Sim workers query the policy over HTTP. Scales to N replicas.

2. **Path B (file-bridge)**: NVIDIA's published G1 pick-and-place fine-tune (`GR00T-N1.6-G1-PnPAppleToPlate`), UNITREE_G1 embodiment, file-bridged to Isaac Lab via separate conda env. Required because N1.6's pinned dependencies conflict with Isaac Sim 5.1's torch pin.

## Demo GIFs

### Path A: Zero-shot N1.7-3B

Base GR00T-3B with REAL_G1 embodiment. The robot moves under policy control, but the base model is not trained on this task distribution so motions are exploratory rather than task-directed.

![N1.7 zero-shot](demos/g1_groot_n17_zeroshot.gif)

### Path B: N1.6 G1 fine-tune

NVIDIA's `GR00T-N1.6-G1-PnPAppleToPlate` checkpoint (the actual G1 pick-place fine-tune) loaded in a dedicated `groot-n16` conda env, file-bridged to Isaac Lab. Inference latency 734ms cold, 154 to 166ms warm on A10G.

![N1.6 G1 fine-tune](demos/g1_groot_n16_g1pnp.gif)

### Path B (polished view)

Same N1.6 GIF, post-processed for clearer presentation (trimmed black frames, tighter crop, brighter, slower playback).

![N1.6 polished](demos/g1_groot_n16_polished.gif)

## Status

| Component | Status | Notes |
|---|---|---|
| Isaac Lab + G1 pick-place scene | Working | NVIDIA pinocchio #4090 workaround required |
| GR00T N1.7-3B inference (REAL_G1) | Working | ~720ms cold, ~80ms warm on A10G |
| GR00T N1.6 G1 fine-tune (UNITREE_G1) | Working | Dedicated `groot-n16` conda env |
| Ray Serve HTTP policy deployment | Working | FastAPI ingress, scales to N replicas |
| End-to-end rollout, Path A | Working | `path_a_ray_serve/single_shot.py` |
| End-to-end rollout, Path B | Working | `path_b_file_bridge/orchestrate_n16.sh` |
| Real joint state ablation | Code committed, run pending | Real robot.data.joint_pos vs zeroed state |
| Parallel sim eval (100 rollouts) | Next iteration | Architecture supports it |

## Architecture

### Path A: Ray Serve HTTP

```
head (CPU)
  +--> Ray Serve cluster
  |      +--> GR00TPolicyServer (REAL_G1) on a GPU worker
  |              POST /predict (FastAPI ingress)
  |              GET  /stats
  |
  +--> N sim worker subprocesses on N GPU workers
         each shell-execs:
           python sim_worker.py --policy-url http://HEAD:8000
```

**Why subprocess instead of Ray actors:** Isaac Sim uses `omni.kit.async_engine` which expects a main asyncio event loop. Inside Ray actor threads, `MainEventLoopWrapper.g_main_event_loop` is None. Scene loading then crashes with `AttributeError: 'NoneType' object has no attribute 'create_task'`. A subprocess gives Isaac Sim a clean Python interpreter and fresh event loop.

### Path B: File-bridge

```
Single GPU worker:

  +-- inference_server.py --+    +-- sim_runner_n16.py --+
  | env: groot-n16          |    | env: base (Isaac Sim) |
  | N1.6 G1 policy          |    | Isaac Lab G1 task     |
  +-- /tmp/bridge/req/  <---+----+- obs pickle ----------+
  |                                                       |
  +-- /tmp/bridge/resp/ -- action chunks --> env.step ----+
```

**Why two envs and a file bridge:** N1.6's pinned dependencies (`torch==2.7.1`, `transformers==4.51.3`, `diffusers==0.35.1`, `flash-attn 2.7.4.post1+cu12torch2.7`) collide with Isaac Sim 5.1's torch pin. Separate conda envs avoid the conflict. Pickle files in `/tmp/bridge/` carry obs and action chunks across the boundary on the same node, so latency stays low.

## Repo layout

```
demos/                          rendered GIFs
path_a_ray_serve/               Ray Serve HTTP architecture (N1.7)
  g1_env.py                     Isaac Lab G1 wrapper, obs/action translation
  policy_server.py              Ray Serve deployment, FastAPI ingress
  sim_worker.py                 standalone subprocess, queries policy via HTTP
  run_demo.py                   orchestrator, fans out N sim workers
  single_shot.py                self-contained end-to-end test (1 process)
path_b_file_bridge/             N1.6 G1 fine-tune via file bridge
  n16_inference_server.py       loads N1.6 policy, watches /tmp/bridge/
  sim_runner_n16.py             Isaac Lab runner, talks to server via files
  orchestrate_n16.sh            launches both processes on one worker
tools/
  polish_gif.py                 post-process GIFs for presentation
  set_token.py                  push HF_TOKEN to all workers
  setup_workers.sh              initial worker bringup
  test_g1_sim.py                Isaac Lab smoke test
  test_groot_standalone.py      GR00T inference smoke test
```

## Required runtime patches

Working with current pip versions against pinned model expectations needed four patches.

**1. VideoInput shim.** In transformers 4.54+, `VideoInput` was moved from `transformers.image_utils` to `transformers.video_utils`. Eagle's dynamic processor (used by GR00T) still imports from the old location.

```python
import transformers.image_utils
from transformers.video_utils import VideoInput
transformers.image_utils.VideoInput = VideoInput
```

**2. flash_attention_2 force.** Qwen3 VLM asserts `_attn_implementation == "flash_attention_2"` but `AutoModel.from_config` does not propagate the `attn_implementation` kwarg through. Monkey-patch `_BaseAutoModelClass.from_config` to inject it.

**3. HF_TOKEN propagation.** Cosmos-Reason2-2B (N1.7's VLM backbone) is a gated model. Worker subprocesses need `HF_TOKEN` in their environment. We propagate via Ray's `runtime_env={"env_vars": {"HF_TOKEN": ...}}`.

**4. Pinocchio pre-import.** NVIDIA IsaacLab issue #4090. Pinocchio's C++ `std::vector<std::string>` binding gets corrupted after Isaac Lab loads a robot URDF. Workaround: `import pinocchio` before `AppLauncher`. Confirmed by NVIDIA on the issue tracker.

## Embodiment and obs/action schemas

### N1.7, EmbodimentTag.REAL_G1

Obs: nested dict.
- `video.ego_view`: `(B, 2, H, W, 3) uint8` (2 frame history)
- `state`: 7 keys including `left_wrist_eef_9d` (3 pos plus 6-element flattened rotation matrix), arms, hands, waist
- `language.annotation.human.task_description`: `[[str]]`

Action: 40-step chunk with 9 keys (left and right wrist, arms, hands, waist, base height command, navigate command).

### N1.6, EmbodimentTag.UNITREE_G1

Obs: nested dict.
- `video.ego_view`: `(B, 1, H, W, 3) uint8` (single frame)
- `state`: 7 full-body keys (`left_leg`, `right_leg`, `waist`, `left_arm`, `right_arm`, `left_hand`, `right_hand`)
- `language`: same as above

Action: 30-step chunk with 7 keys (upper body and waist, plus base_height_command, navigate_command).

### Isaac Lab task action mapping

`Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0` accepts actions of shape `(1, 28)`. We pack the policy output as `left_arm(7) + right_arm(7) + left_hand(7) + right_hand(7) = 28`.

## Joint index mapping (Isaac Lab 43-DOF G1 to N1.6 schema)

```
left_leg [6]:   [0, 3, 6, 9, 13, 17]    hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
right_leg [6]:  [1, 4, 7, 10, 14, 18]
waist [3]:      [2, 5, 8]                yaw, roll, pitch
left_arm [7]:   [11, 15, 19, 21, 23, 25, 27]   shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
right_arm [7]:  [12, 16, 20, 22, 24, 26, 28]
left_hand [7]:  [29, 35, 30, 36, 31, 37, 41]   index 0+1, middle 0+1, thumb 0+1+2
right_hand [7]: [32, 38, 33, 39, 34, 40, 42]
```

## Setup

```bash
# initial worker bringup (Path A: Ray Serve, N1.7)
bash tools/setup_workers.sh

# Hugging Face login (must accept Cosmos-Reason2-2B terms first)
hf auth login

# distribute HF_TOKEN across workers
python tools/set_token.py
```

For Path B, the `groot-n16` conda env with N1.6's pinned deps must be built on the worker that will run inference. See README in `path_b_file_bridge/` (or replicate the steps from `n16_inference_server.py`).

## Run

```bash
# Path A: single rollout, N1.7-3B base
python path_a_ray_serve/single_shot.py

# Path A: full Ray Serve fan-out (placeholder, no GR00T)
python path_a_ray_serve/run_demo.py --placeholder --num-workers 2 --episodes 1

# Path A: full Ray Serve with GR00T inference
python path_a_ray_serve/run_demo.py --num-workers 2 --episodes 1

# Path B: N1.6 G1 fine-tune
bash path_b_file_bridge/orchestrate_n16.sh
```

## Known limitations

The current Path B GIF was produced with zeroed joint state inputs (the model only sees camera and language). Wiring real joint positions from `robot.data.joint_pos[0]` into the obs is the natural next iteration for better task success. The joint mapping is documented above.

NVIDIA trained the G1 fine-tune on a slightly different scene layout than Isaac Lab's `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0`. Out-of-distribution shift between training and eval scenes degrades zero-shot performance.

A10G GPUs (24GB) work for inference of all checkpoints used here, but training GR00T from scratch is not feasible on this hardware (NVIDIA recommends H100). Fine-tuning on A10G is borderline and would benefit from LoRA or partial fine-tuning of just the action head.

## Next iterations

1. Real-state ablation, Path B: feed actual joint positions instead of zeros (joint mapping is implemented in `sim_runner_n16_v3.py` from session notes).
2. Parallel sim eval: 100 rollouts across N workers, success rate and throughput numbers.
3. Action horizon tuning: Path B currently executes 8 to 16 steps per 30-step chunk. Search for stable values.
4. LoRA fine-tune on Isaac Lab demonstrations to close distribution gap.

## Acknowledgments

- NVIDIA IsaacLab team for the [pinocchio #4090 workaround](https://github.com/isaac-sim/IsaacLab/issues/4090)
- NVIDIA Isaac-GR00T `n1.6-release` and `main` branches
- Anyscale for the cluster
