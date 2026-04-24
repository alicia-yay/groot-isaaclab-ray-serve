"""
Standalone GR00T inference test.

Run on a worker with a GPU BEFORE trying run_demo.py. This confirms:
  1. GR00T's uv env is installed correctly
  2. The model weights download from HF
  3. Gr00tPolicy loads without error
  4. get_action() returns something sensibly shaped

If this fails, fix GR00T install first - adding Ray Serve on top won't help.

Usage:
    # Test base GR00T N1.6 on the GR1 demo data (comes with the repo):
    python test_groot_standalone.py

    # Test G1 fine-tune:
    python test_groot_standalone.py --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \\
        --embodiment-tag UNITREE_G1 --data-config unitree_g1
"""
import argparse
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--embodiment-tag", default="GR1")
    parser.add_argument("--data-config", default="gr1_arms_only")
    parser.add_argument("--denoising-steps", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading {args.model_path} (embodiment={args.embodiment_tag})...")
    t0 = time.time()

    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import load_data_config

    data_config = load_data_config(args.data_config)

    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )

    print(f"Loaded in {time.time() - t0:.1f}s")
    print(f"Modality config keys: {list(data_config.modality_config().keys())}")

    # Build a minimal dummy observation.  The exact keys MUST match what
    # data_config.modality_config() declares - inspect it above.
    if args.embodiment_tag == "UNITREE_G1":
        obs = {
            "video.ego_view":    np.zeros((1, 224, 224, 3), dtype=np.uint8),
            "state.left_arm":    np.zeros((1, 7),  dtype=np.float32),
            "state.right_arm":   np.zeros((1, 7),  dtype=np.float32),
            "state.left_hand":   np.zeros((1, 6),  dtype=np.float32),
            "state.right_hand":  np.zeros((1, 6),  dtype=np.float32),
            "state.waist":       np.zeros((1, 3),  dtype=np.float32),
            "annotation.human.task_description": ["pick up the apple"],
        }
    else:  # GR1 default
        obs = {
            "video.ego_view":    np.zeros((1, 224, 224, 3), dtype=np.uint8),
            "state.left_arm":    np.zeros((1, 7),  dtype=np.float32),
            "state.right_arm":   np.zeros((1, 7),  dtype=np.float32),
            "state.left_hand":   np.zeros((1, 6),  dtype=np.float32),
            "state.right_hand":  np.zeros((1, 6),  dtype=np.float32),
            "annotation.human.task_description": ["pick up the object"],
        }

    print(f"\nCalling get_action(...) 5 times to measure steady-state latency")
    for i in range(5):
        t0 = time.time()
        action = policy.get_action(obs)
        dt_ms = (time.time() - t0) * 1000
        print(f"  Call {i}: {dt_ms:.1f}ms")

    print(f"\nAction keys: {list(action.keys())}")
    for k, v in action.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")

    print("\nGR00T standalone test passed.")


if __name__ == "__main__":
    main()
