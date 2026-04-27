"""
Single-process end-to-end test:
  Isaac Lab G1 sim + GR00T N1.7-3B inference + GIF save, all in one Python process
  on one Ray worker. No Ray Serve. No subprocess. No HTTP.

Goal: prove the full pipeline is alive. If this works, we scale.
"""
import sys, os, time
import numpy as np
sys.stdout.reconfigure(line_buffering=True)


def main():
    # ============================================================
    # 1. HF login
    # ============================================================
    from huggingface_hub import login
    tok = os.environ.get("HF_TOKEN")
    if tok:
        login(token=tok, add_to_git_credential=False)
        print(f"[1] HF login OK", flush=True)

    # ============================================================
    # 2. Patches (verified working last session)
    # ============================================================
    import transformers.image_utils
    if not hasattr(transformers.image_utils, "VideoInput"):
        from transformers.video_utils import VideoInput
        transformers.image_utils.VideoInput = VideoInput
    print("[2] VideoInput shim", flush=True)

    from transformers.models.auto.auto_factory import _BaseAutoModelClass
    _orig = _BaseAutoModelClass.from_config.__func__

    def _p(cls, config, **kwargs):
        if hasattr(config, "text_config"):
            config.text_config._attn_implementation = "flash_attention_2"
        config._attn_implementation = "flash_attention_2"
        if "attn_implementation" not in kwargs:
            kwargs["attn_implementation"] = "flash_attention_2"
        return _orig(cls, config, **kwargs)

    _BaseAutoModelClass.from_config = classmethod(_p)
    print("[3] flash_attn forced", flush=True)

    # ============================================================
    # 3. Boot Isaac Lab BEFORE GR00T (because pinocchio workaround)
    # ============================================================
    print("[4] importing pinocchio (NVIDIA #4090 workaround)", flush=True)
    import pinocchio
    from isaaclab.app import AppLauncher
    print("[5] launching Isaac Sim (~60-90s)", flush=True)
    t0 = time.time()
    app = AppLauncher(headless=True, enable_cameras=True).app
    print(f"[5] Isaac Sim booted in {time.time() - t0:.1f}s", flush=True)

    # ============================================================
    # 4. Make G1 env
    # ============================================================
    print("[6] creating G1 env", flush=True)
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.manager_based.locomanipulation import pick_place  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    import torch

    task = "Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0"
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=1, use_fabric=True)
    env = gym.make(task, cfg=cfg, render_mode="rgb_array")
    print(f"[7] env action_space: {env.action_space}", flush=True)
    print(f"[7] env observation_space (top-level): {list(env.observation_space.keys())}", flush=True)

    obs, info = env.reset(seed=42)
    print(f"[8] reset OK; obs keys: {list(obs.keys())}", flush=True)
    if "policy" in obs:
        print(f"    obs['policy'] keys: {list(obs['policy'].keys())[:20]}", flush=True)

    # Render one frame to confirm camera works
    print("[9] rendering one frame...", flush=True)
    frame = env.render()
    print(f"[9] frame shape: {np.asarray(frame).shape if frame is not None else None}", flush=True)

    # ============================================================
    # 5. Load GR00T N1.7-3B (REAL_G1)
    # ============================================================
    print("[10] loading GR00T N1.7-3B...", flush=True)
    t0 = time.time()
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.REAL_G1,
        model_path="nvidia/GR00T-N1.7-3B",
        device="cuda:0",
    )
    print(f"[10] loaded in {time.time() - t0:.1f}s", flush=True)

    # ============================================================
    # 6. Build a GR00T-format obs from the rendered frame + identity wrist
    # ============================================================
    print("[11] building GR00T obs", flush=True)
    identity_pose = np.array([0.3, 0.0, 0.2, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    rgb = np.asarray(frame, dtype=np.uint8) if frame is not None \
        else np.zeros((480, 640, 3), dtype=np.uint8)
    if rgb.ndim == 3:
        rgb = rgb[None, ...]  # add batch
    if rgb.ndim == 4:
        rgb = rgb[:, None]   # add time
    rgb = np.broadcast_to(rgb, (1, 2, *rgb.shape[2:])).astype(np.uint8).copy()  # T=2

    groot_obs = {
        "video": {"ego_view": rgb},
        "state": {
            "left_wrist_eef_9d":  identity_pose[None, None, :].copy(),
            "right_wrist_eef_9d": identity_pose[None, None, :].copy(),
            "left_hand":   np.zeros((1, 1, 7), dtype=np.float32),
            "right_hand":  np.zeros((1, 1, 7), dtype=np.float32),
            "left_arm":    np.zeros((1, 1, 7), dtype=np.float32),
            "right_arm":   np.zeros((1, 1, 7), dtype=np.float32),
            "waist":       np.zeros((1, 1, 3), dtype=np.float32),
        },
        "language": {
            "annotation.human.task_description": [["pick up the apple and place it on the plate"]],
        },
    }

    # ============================================================
    # 7. Get one action chunk from GR00T
    # ============================================================
    print("[12] querying GR00T...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        action_chunk, _info = policy.get_action(groot_obs)
    print(f"[12] action_chunk in {(time.time()-t0)*1000:.0f}ms; keys: {list(action_chunk.keys())}", flush=True)

    # ============================================================
    # 8. Step Isaac Lab using GR00T's action - N steps before re-querying
    # ============================================================
    NUM_QUERIES = 4       # number of GR00T calls
    STEPS_PER_QUERY = 8   # how many actions to use per chunk
    SAVE_EVERY = 1        # save every frame

    frames = [rgb[0, 0]]  # initial frame
    t_total = time.time()
    for q in range(NUM_QUERIES):
        if q > 0:
            with torch.no_grad():
                action_chunk, _ = policy.get_action(groot_obs)
        for s in range(STEPS_PER_QUERY):
            # GR00T outputs (1, 40, D). Take step s. Concatenate left_arm/right_arm/left_hand/right_hand into 28 dims.
            la = np.asarray(action_chunk["left_arm"])[0, s]    # (7,)
            ra = np.asarray(action_chunk["right_arm"])[0, s]   # (7,)
            lh = np.asarray(action_chunk["left_hand"])[0, s]   # (7,)
            rh = np.asarray(action_chunk["right_hand"])[0, s]  # (7,)
            flat = np.concatenate([la, ra, lh, rh], axis=-1)[None, :]  # (1, 28)
            action_t = torch.as_tensor(flat, dtype=torch.float32)
            obs, reward, terminated, truncated, info = env.step(action_t)

            if (q * STEPS_PER_QUERY + s) % SAVE_EVERY == 0:
                f = env.render()
                if f is not None:
                    frames.append(np.asarray(f, dtype=np.uint8))

            if terminated or truncated:
                print(f"[13] episode ended at q={q} s={s}", flush=True)
                break

        # Update obs for next query - just refresh video, leave state zeros (still zero-shot anyway)
        new_frame = env.render()
        if new_frame is not None:
            new_rgb = np.asarray(new_frame, dtype=np.uint8)[None, None]
            new_rgb = np.broadcast_to(new_rgb, (1, 2, *new_rgb.shape[2:])).astype(np.uint8).copy()
            groot_obs["video"]["ego_view"] = new_rgb
        print(f"  query {q+1}/{NUM_QUERIES} done; total frames: {len(frames)}", flush=True)

    total_time = time.time() - t_total
    print(f"[14] {NUM_QUERIES * STEPS_PER_QUERY} steps in {total_time:.1f}s", flush=True)

    # ============================================================
    # 9. Save GIF
    # ============================================================
    print("[15] saving GIF...", flush=True)
    import imageio
    out_path = "/home/ray/groot_demo/demo_output/g1_groot_n17_zeroshot.gif"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, frames, fps=15, loop=0)
    print(f"[15] SAVED: {out_path} ({len(frames)} frames)", flush=True)

    env.close()
    print("SUCCESS", flush=True)


if __name__ == "__main__":
    main()
