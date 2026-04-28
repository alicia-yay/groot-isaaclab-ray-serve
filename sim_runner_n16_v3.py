"""
v3: real joint state + smaller chunks + camera resize + action noise
"""
import sys, os, time, pickle, uuid
import numpy as np
sys.stdout.reconfigure(line_buffering=True)

LEFT_LEG_IDX   = [0, 3, 6, 9, 13, 17]
RIGHT_LEG_IDX  = [1, 4, 7, 10, 14, 18]
WAIST_IDX      = [2, 5, 8]
LEFT_ARM_IDX   = [11, 15, 19, 21, 23, 25, 27]
RIGHT_ARM_IDX  = [12, 16, 20, 22, 24, 26, 28]
LEFT_HAND_IDX  = [29, 35, 30, 36, 31, 37, 41]
RIGHT_HAND_IDX = [32, 38, 33, 39, 34, 40, 42]


def query_policy(obs, timeout=60.0):
    req_id = uuid.uuid4().hex[:8]
    req_path = f'/tmp/bridge/req/{req_id}.pkl'
    resp_path = f'/tmp/bridge/resp/{req_id}.pkl'
    err_path = f'/tmp/bridge/resp/{req_id}.err'
    with open(f'{req_path}.tmp', 'wb') as f: pickle.dump(obs, f)
    os.rename(f'{req_path}.tmp', req_path)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(resp_path):
            with open(resp_path, 'rb') as f: resp = pickle.load(f)
            os.remove(resp_path)
            return resp
        if os.path.exists(err_path):
            with open(err_path) as f: err = f.read()
            os.remove(err_path)
            raise RuntimeError(f'Server error: {err}')
        time.sleep(0.05)
    raise TimeoutError(f'No response after {timeout}s')


def wait_for_server_ready(timeout=240.0):
    print('[sim] waiting for server READY...', flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists('/tmp/bridge/READY'):
            print(f'[sim] server ready ({time.time()-t0:.1f}s)', flush=True)
            return
        time.sleep(1.0)
    raise TimeoutError('server never became ready')


def resize_to_224(rgb):
    """Crop+resize a (H, W, 3) frame to (224, 224, 3) - rough match to typical training distribution."""
    if rgb is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    rgb = np.asarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    # Center crop to square
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    cropped = rgb[y0:y0+side, x0:x0+side]
    # Simple nearest-neighbor downsample (no PIL/cv2 dependency)
    if side != 224:
        idxs = (np.linspace(0, side-1, 224)).astype(np.int32)
        cropped = cropped[idxs][:, idxs]
    return cropped


def build_obs(frame_full, jp_np, instruction):
    """Build N1.6 obs with REAL joint state and 224x224 image."""
    rgb_small = resize_to_224(frame_full)
    rgb = rgb_small[None, None]  # (1, 1, 224, 224, 3)
    
    def _slice(idx):
        return jp_np[idx][None, None].astype(np.float32)
    
    return {
        'video': {'ego_view': rgb},
        'state': {
            'left_leg':   _slice(LEFT_LEG_IDX),
            'right_leg':  _slice(RIGHT_LEG_IDX),
            'waist':      _slice(WAIST_IDX),
            'left_arm':   _slice(LEFT_ARM_IDX),
            'right_arm':  _slice(RIGHT_ARM_IDX),
            'left_hand':  _slice(LEFT_HAND_IDX),
            'right_hand': _slice(RIGHT_HAND_IDX),
        },
        'language': {
            'annotation.human.task_description': [[instruction]],
        },
    }


def main():
    wait_for_server_ready()
    
    print('[sim] booting Isaac Lab...', flush=True)
    import pinocchio
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app
    print('[sim] Isaac Sim booted', flush=True)
    
    import gymnasium as gym
    import isaaclab_tasks
    from isaaclab_tasks.manager_based.locomanipulation import pick_place
    from isaaclab_tasks.utils import parse_env_cfg
    import torch
    
    task = "Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0"
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=1, use_fabric=True)
    env = gym.make(task, cfg=cfg, render_mode="rgb_array")
    print(f'[sim] action_space: {env.action_space}', flush=True)
    
    obs, info = env.reset(seed=42)
    print(f'[sim] reset OK', flush=True)
    
    robot = env.unwrapped.scene['robot']
    def get_jp():
        return robot.data.joint_pos[0].detach().cpu().numpy()
    
    # v3 hyperparameters
    NUM_QUERIES = 16
    STEPS_PER_QUERY = 6     # Smaller chunks - more frequent state updates
    ACTION_NOISE_STD = 0.01  # Small Gaussian noise on actions
    INSTRUCTION = "pick up the apple and place it on the plate"
    
    # Capture FULL-RES frames for the GIF (don't show 224x224 to viewer)
    frames = []
    full_frame = env.render()
    if full_frame is not None:
        frames.append(np.asarray(full_frame, dtype=np.uint8))
    
    print(f'[sim] starting {NUM_QUERIES} queries x {STEPS_PER_QUERY} = {NUM_QUERIES*STEPS_PER_QUERY} total steps', flush=True)
    print(f'[sim] action_noise_std={ACTION_NOISE_STD}', flush=True)
    
    for q in range(NUM_QUERIES):
        jp = get_jp()
        full_frame = env.render()
        obs_dict = build_obs(full_frame, jp, INSTRUCTION)
        
        t0 = time.time()
        try:
            resp = query_policy(obs_dict, timeout=120.0)
        except Exception as e:
            print(f'[sim] q{q}: query failed: {e}', flush=True)
            break
        action_chunk = resp['action']
        
        if q == 0:
            print(f'[sim] q0: {(time.time()-t0)*1000:.0f}ms', flush=True)
            for k, v in action_chunk.items():
                v = np.asarray(v)
                print(f'    action.{k}: {v.shape}', flush=True)
        elif q % 4 == 0:
            print(f'[sim] q{q}: {(time.time()-t0)*1000:.0f}ms', flush=True)
        
        # Execute STEPS_PER_QUERY actions
        for s in range(STEPS_PER_QUERY):
            la = np.asarray(action_chunk['left_arm'])[0, s]
            ra = np.asarray(action_chunk['right_arm'])[0, s]
            lh = np.asarray(action_chunk['left_hand'])[0, s]
            rh = np.asarray(action_chunk['right_hand'])[0, s]
            flat = np.concatenate([la, ra, lh, rh], axis=-1)[None, :]
            
            # Add small noise
            if ACTION_NOISE_STD > 0:
                noise = np.random.randn(*flat.shape).astype(np.float32) * ACTION_NOISE_STD
                flat = flat + noise
            
            action_t = torch.as_tensor(flat, dtype=torch.float32)
            obs, reward, terminated, truncated, info = env.step(action_t)
            
            f = env.render()
            if f is not None:
                frames.append(np.asarray(f, dtype=np.uint8))
            
            if terminated or truncated:
                print(f'[sim] episode end q={q} s={s} reward={float(reward):.3f}', flush=True)
                break
        else:
            continue
        break
    
    print(f'[sim] {len(frames)} frames', flush=True)
    
    import imageio
    out_path = '/home/ray/groot_demo/demo_output/g1_groot_n16_v3.gif'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, frames, fps=15, loop=0)
    print(f'[sim] SAVED: {out_path} ({len(frames)} frames)', flush=True)
    
    with open('/tmp/bridge/STOP', 'w') as f: f.write('done')
    env.close()
    print('SUCCESS', flush=True)


if __name__ == '__main__':
    main()
