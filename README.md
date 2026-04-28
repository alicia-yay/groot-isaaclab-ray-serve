# GR00T + Isaac Lab + G1 + Ray Serve Demo

End-to-end infrastructure for serving NVIDIA GR00T VLA models to Isaac Lab Unitree G1 simulations on an Anyscale Ray cluster.

## Demo GIFs

### Zero-shot N1.7-3B base model
GR00T-N1.7-3B with `REAL_G1` embodiment, no fine-tuning. The robot is in the pick-place scene with a real policy driving it, but the base model isn't trained on this task distribution so motions are exploratory.

![N1.7 zero-shot](g1_groot_n17_zeroshot.gif)

### N1.6 G1 fine-tune (`GR00T-N1.6-G1-PnPAppleToPlate`)
The actual NVIDIA-published G1 pick-and-place fine-tune, loaded in a dedicated conda env (file-bridged to Isaac Lab to avoid torch version conflicts with Isaac Sim 5.1).

![N1.6 G1 fine-tune](g1_groot_n16_g1pnp.gif)

## Status

| Component | Status | Notes |
|---|---|---|
| Isaac Lab + G1 pick-place scene | Working | NVIDIA pinocchio #4090 workaround |
| GR00T N1.7-3B inference (REAL_G1) | Working | ~720ms cold / ~80ms warm on A10G |
| GR00T N1.6 G1 fine-tune (UNITREE_G1) | Working | Dedicated `groot-n16` conda env, file-bridge to Isaac Lab |
| Ray Serve HTTP policy deployment | Working | FastAPI ingress, scales to N replicas |
| End-to-end rollout + GIF (zero-shot) | Working | `single_shot.py` |
| End-to-end rollout + GIF (G1 fine-tune) | Working | `orchestrate_n16.sh` |

## Architecture

Two demo paths — both real:

### Path A: Ray Serve HTTP (N1.7-3B)

```
head ─┬─→ Ray Serve cluster ─→ GR00TPolicyServer (REAL_G1) on a GPU worker
      │                         POST /predict (FastAPI ingress)
      │
      └─→ Sim worker subprocesses on N GPUs
            each shell-execs: python sim_worker.py --policy-url http://HEAD:8000
```

### Path B: File-bridge (N1.6 G1 fine-tune)

```
Single GPU worker:
  ┌─ inference_server.py ──┐    ┌── sim_runner_n16.py ──┐
  │ env: groot-n16          │    │ env: base (Isaac Sim) │
  │ N1.6 G1 policy          │    │ Isaac Lab G1 task     │
  └─ /tmp/bridge/req/ ←─────┴────┴───── obs pickle ──────┘
   /tmp/bridge/resp/ ──→ action chunks ──→ env.step
```

Why file-bridge for N1.6: N1.6's pinned deps (`torch==2.7.1`, `transformers==4.51.3`, `diffusers==0.35.1`) collide with Isaac Sim 5.1's pins. Separate conda envs avoid the conflict; pickle files in `/tmp/bridge/` carry obs/actions across the boundary.

## Files

**Path A (Ray Serve, N1.7):**
- `g1_env.py` — Isaac Lab G1 wrapper. Includes pinocchio pre-import and obs/action translation for `REAL_G1` schema.
- `policy_server.py` — Ray Serve `Gr00tPolicy` deployment. FastAPI ingress, runtime patches for transformers 4.57.6 vs N1.7's pinned 4.51.3.
- `sim_worker.py` — Standalone subprocess (not Ray actor) talks to policy via HTTP.
- `run_demo.py` — Orchestrator: Ray Serve startup + worker subprocess fan-out.
- `single_shot.py` — Self-contained end-to-end test (one Ray task, no Serve, no subprocess). Produces `g1_groot_n17_zeroshot.gif`.

**Path B (file-bridge, N1.6):**
- `n16_inference_server.py` — Loads `nvidia/GR00T-N1.6-G1-PnPAppleToPlate` with `UNITREE_G1` tag. Watches `/tmp/bridge/req/`, writes to `resp/`.
- `sim_runner_n16.py` — Boots Isaac Lab, queries server via files, saves GIF. Produces `g1_groot_n16_g1pnp.gif`.
- `orchestrate_n16.sh` — Launches both processes on the same worker.

**Setup:**
- `setup_workers.sh` — Initial worker bringup (Isaac-GR00T install, flash-attn prebuilt, conda patches).

## Required runtime patches

Working with these specific versions required four patches:

1. **VideoInput shim**: In transformers ≥4.54, `VideoInput` was moved from `transformers.image_utils` to `transformers.video_utils`. Eagle's dynamic processor (used by GR00T) still imports from the old location.
2. **flash_attention_2 force**: Qwen3 VLM asserts `_attn_implementation == "flash_attention_2"` but `AutoModel.from_config` doesn't propagate `attn_implementation` kwarg through. Monkey-patch `_BaseAutoModelClass.from_config`.
3. **HF_TOKEN propagation**: Cosmos-Reason2-2B (N1.7's VLM backbone) is gated. Workers need `HF_TOKEN` via Ray `runtime_env={"env_vars": {"HF_TOKEN": ...}}`.
4. **Pinocchio pre-import**: NVIDIA IsaacLab issue #4090. Pinocchio's C++ `std::vector<std::string>` binding gets corrupted after Isaac Lab loads a robot URDF. Workaround: `import pinocchio` before `AppLauncher`. Confirmed by NVIDIA on the issue tracker.

## Embodiment + obs/action schemas

### N1.7 — `EmbodimentTag.REAL_G1`
Obs: nested dict with `video.ego_view (B, 2, H, W, 3) uint8`, 7 state keys including `left_wrist_eef_9d` (9-DOF wrist pose), `language.annotation.human.task_description: [[str]]`.

Action: 40-step chunk, 9 keys (left/right wrist + arm + hand, waist, base_height, navigate).

### N1.6 — `EmbodimentTag.UNITREE_G1`
Obs: nested dict with `video.ego_view (B, 1, H, W, 3) uint8`, 7 full-body state keys (`left_leg`/`right_leg` + waist + arms + hands), language same.

Action: 30-step chunk, 7 keys (upper body only — useful for fixed-base tasks).

Isaac Lab task `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0` accepts `(1, 28)` actions. We pack `left_arm(7) + right_arm(7) + left_hand(7) + right_hand(7) = 28`.

## Known limitations / future work

- **Zero-state vs real-state for N1.6**: First N1.6 GIF used zeroed joint state (just camera + language). Wiring real joint positions from Isaac Lab into the obs (`robot.data.joint_pos[0]`) is the natural next iteration for better task success.
- **Action horizon**: We execute 8-16 steps from each 30-step chunk. Tuning this for stability is task-specific.
- **Camera distribution shift**: NVIDIA's training scene differs slightly from Isaac Lab's exact pick-place layout. Out-of-distribution conditions degrade zero-shot performance.

## Run

```bash
# Path A: Ray Serve, N1.7-3B base, single rollout
python single_shot.py

# Path A: Ray Serve, N1.7-3B base, fan-out (placeholder for arch testing)
python run_demo.py --placeholder --num-workers 2 --episodes 1

# Path A: full GR00T inference with sim
python run_demo.py --num-workers 2 --episodes 1

# Path B: N1.6 G1 fine-tune
bash orchestrate_n16.sh
```

## Acknowledgments

- NVIDIA IsaacLab team for the [pinocchio #4090 workaround](https://github.com/isaac-sim/IsaacLab/issues/4090)
- NVIDIA Isaac-GR00T `n1.6-release` and `main` branches
- Anyscale for the cluster
