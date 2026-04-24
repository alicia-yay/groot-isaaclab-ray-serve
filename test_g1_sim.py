"""
Smoke test: one G1 worker, random actions, 100 steps, saves GIF.

Run this FIRST to confirm:
  1. Isaac Lab launches in this environment
  2. The G1 task registers and resets cleanly
  3. action_space is shaped as expected
  4. rendering produces usable frames

No Ray, no Ray Serve, no GR00T. If this fails, nothing else will work.

Usage:
    python test_g1_sim.py

Expected output:
    - Prints obs_space and action_space
    - Runs 100 sim steps with random actions
    - Saves test_g1_sim.gif
"""
import os
import numpy as np


def main():
    from g1_env import G1LocomanipulationEnv

    env = G1LocomanipulationEnv(
        task_name="Isaac-PickPlace-Locomanipulation-G1-Abs-v0",
        language_instruction="pick up the apple",
        headless=True,
        seed=0,
    )

    print("Resetting env...")
    obs = env.reset()
    print(f"Observation keys: {list(obs.keys())}")
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")

    # Figure out an action shape by flattening a zero dict against env.step.
    # Since we don't know the exact split yet, just sample from the space.
    raw_action_space = env.env.action_space
    print(f"\nAction space: {raw_action_space}")

    frames = []
    n_steps = 100
    print(f"\nRunning {n_steps} steps with random actions...")
    for i in range(n_steps):
        # Sample directly from the raw env action space (bypass the GR00T
        # chunk format since we're not using a policy here).
        raw_action = raw_action_space.sample()
        # env.step expects a chunk-dict; we short-circuit and call the raw env.
        obs_raw, reward, terminated, truncated, info = env.env.step(raw_action)
        if i % 2 == 0:
            frames.append(env.render_frame())
        if terminated or truncated:
            print(f"  Episode ended at step {i}")
            break

    gif_path = "test_g1_sim.gif"
    try:
        import imageio
        imageio.mimsave(gif_path, frames, fps=15, loop=0)
        print(f"\nSaved {len(frames)} frames to {gif_path}")
    except ImportError:
        print("\nimageio not installed - skipping GIF save. `pip install imageio`")

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
