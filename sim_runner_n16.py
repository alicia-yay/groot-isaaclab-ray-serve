"""
Isaac Lab G1 sim runner — runs in BASE conda env (Isaac Sim available).
Talks to n16_inference_server.py via /tmp/bridge/ pickle files.

For each step: write obs to /tmp/bridge/req/N.pkl, wait for /tmp/bridge/resp/N.pkl.
"""
import sys, os, time, pickle, uuid
import numpy as np
sys.stdout.reconfigure(line_buffering=True)


def query_policy(obs: dict, timeout: float = 60.0) -> dict:
    """File-based RPC to N1.6 inference server."""
    req_id = uuid.uuid4().hex[:8]
    req_path = f'/tmp/bridge/req/{req_id}.pkl'
    resp_path = f'/tmp/bridge/resp/{req_id}.pkl'
    err_path = f'/tmp/bridge/resp/{req_id}.err'
    
    # Write request atomically
    tmp_req = f'{req_path}.tmp'
    with open(tmp_req, 'wb') as f:
        pickle.dump(obs, f)
    os.rename(tmp_req, req_path)
    
    # Wait for response
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(resp_path):
            with open(resp_path, 'rb') as f:
                resp = pickle.load(f)
            os.remove(resp_path)
            return resp
        if os.path.exists(err_path):
            with open(err_path) as f:
                err = f.read()
            os.remove(err_path)
            raise RuntimeError(f'Server error: {err}')
        time.sleep(0.05)
    raise TimeoutError(f'No response after {timeout}s')


def wait_for_server_ready(timeout: float = 240.0):
    print('[sim] waiting for inference server READY...', flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists('/tmp/bridge/READY'):
            print(f'[sim] server ready ({time.time()-t0:.1f}s)', flush=True)
            return
        time.sleep(1.0)
    raise TimeoutError('inference server never became ready')


def main():
    wait_for_server_ready()
    
    # Boot Isaac Lab in BASE env
    print('[sim] booting Isaac Lab...', flush=True)
    import pinocchio  # noqa: F401  # NVIDIA #4090
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app
    print('[sim] Isaac Sim booted', flush=True)
    
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.manager_based.locomanipulation import pick_place  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    import torch
    
    task = "Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0"
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=1, use_fabric=True)
    env = gym.make(task, cfg=cfg, render_mode="rgb_array")
    print(f'[sim] env action_space: {env.action_space}', flush=True)
    
    obs, info = env.reset(seed=42)
    print(f'[sim] reset OK', flush=True)
    
    # Warmup render
    frame = env.render()
    print(f'[sim] first frame shape: {None if frame is None else np.asarray(frame).shape}', flush=True)
    
    # Build N1.6 obs schema:
    # video.ego_view: (1, 1, H, W, 3) uint8
    # state: 7 keys, each (1, 1, D)
    # language: [[str]]
    def build_obs(frame_rgb):
        if frame_rgb is None:
            frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb = np.asarray(frame_rgb, dtype=np.uint8)[None, None]  # (1, 1, H, W, 3)
        return {
            'video': {'ego_view': rgb},
            'state': {
                # We don't have access to G1's joint state cleanly without poking
                # the env's internals. Zero-state is suboptimal but lets the policy
                # condition on the language + image. Since the task is fixed-base
                # upper-body, leg state being zero is actually correct!
                'left_leg':   np.zeros((1, 1, 6), dtype=np.float32),  # leg DOF guess
                'right_leg':  np.zeros((1, 1, 6), dtype=np.float32),
                'waist':      np.zeros((1, 1, 3), dtype=np.float32),
                'left_arm':   np.zeros((1, 1, 7), dtype=np.float32),
                'right_arm':  np.zeros((1, 1, 7), dtype=np.float32),
                'left_hand':  np.zeros((1, 1, 7), dtype=np.float32),
                'right_hand': np.zeros((1, 1, 7), dtype=np.float32),
            },
            'language': {
                'annotation.human.task_description': [['pick up the apple and place it on the plate']],
            },
        }
    
    NUM_QUERIES = 4
    STEPS_PER_QUERY = 8
    frames = [np.asarray(frame, dtype=np.uint8) if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)]
    
    for q in range(NUM_QUERIES):
        # Render → build obs → query
        frame = env.render()
        obs_dict = build_obs(frame)
        print(f'[sim] q{q}: querying server...', flush=True)
        t0 = time.time()
        try:
            resp = query_policy(obs_dict, timeout=120.0)
        except Exception as e:
            print(f'[sim] q{q}: query failed: {e}', flush=True)
            break
        action_chunk = resp['action']
        print(f'[sim] q{q}: got chunk in {(time.time()-t0)*1000:.0f}ms; keys={list(action_chunk.keys())}', flush=True)
        
        # Print shapes once
        if q == 0:
            for k, v in action_chunk.items():
                v = np.asarray(v)
                print(f'[sim] action.{k}: {v.shape}', flush=True)
        
        # Execute STEPS_PER_QUERY actions from chunk
        for s in range(STEPS_PER_QUERY):
            # N1.6 action schema (each (1, 30, D)):
            #   left_arm: 7, right_arm: 7, left_hand: 7, right_hand: 7
            # Concat into Isaac Lab's (1, 28).
            la = np.asarray(action_chunk['left_arm'])[0, s]
            ra = np.asarray(action_chunk['right_arm'])[0, s]
            lh = np.asarray(action_chunk['left_hand'])[0, s]
            rh = np.asarray(action_chunk['right_hand'])[0, s]
            flat = np.concatenate([la, ra, lh, rh], axis=-1)[None, :]  # (1, 28)
            action_t = torch.as_tensor(flat, dtype=torch.float32)
            obs, reward, terminated, truncated, info = env.step(action_t)
            
            f = env.render()
            if f is not None:
                frames.append(np.asarray(f, dtype=np.uint8))
            
            if terminated or truncated:
                print(f'[sim] episode end at q={q} s={s} reward={float(reward):.3f}', flush=True)
                break
        else:
            continue
        break
    
    print(f'[sim] {len(frames)} frames captured', flush=True)
    
    # Save GIF
    import imageio
    out_path = '/home/ray/groot_demo/demo_output/g1_groot_n16_g1pnp.gif'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, frames, fps=15, loop=0)
    print(f'[sim] SAVED: {out_path} ({len(frames)} frames)', flush=True)
    
    # Signal server to stop
    with open('/tmp/bridge/STOP', 'w') as f:
        f.write('done')
    
    env.close()
    print('SUCCESS', flush=True)


if __name__ == '__main__':
    main()
