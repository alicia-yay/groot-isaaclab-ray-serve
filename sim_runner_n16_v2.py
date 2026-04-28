"""
Isaac Lab G1 sim runner v2 — wires REAL joint state into N1.6 obs.
Talks to n16_inference_server.py via /tmp/bridge/ pickle files.

Joint mapping (Isaac Lab 43-DOF G1 → N1.6 schema):
  left_leg [6]:   [0, 3, 6, 9, 13, 17]   (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
  right_leg [6]:  [1, 4, 7, 10, 14, 18]
  waist [3]:      [2, 5, 8]              (yaw, roll, pitch)
  left_arm [7]:   [11, 15, 19, 21, 23, 25, 27]   (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
  right_arm [7]:  [12, 16, 20, 22, 24, 26, 28]
  left_hand [7]:  [29, 35, 30, 36, 31, 37, 41]   (idx0,idx1, mid0,mid1, thmb0,thmb1, thmb2)
  right_hand [7]: [32, 38, 33, 39, 34, 40, 42]
"""
import sys, os, time, pickle, uuid
import numpy as np
sys.stdout.reconfigure(line_buffering=True)


# Joint index mappings
LEFT_LEG_IDX   = [0, 3, 6, 9, 13, 17]
RIGHT_LEG_IDX  = [1, 4, 7, 10, 14, 18]
WAIST_IDX      = [2, 5, 8]
LEFT_ARM_IDX   = [11, 15, 19, 21, 23, 25, 27]
RIGHT_ARM_IDX  = [12, 16, 20, 22, 24, 26, 28]
LEFT_HAND_IDX  = [29, 35, 30, 36, 31, 37, 41]
RIGHT_HAND_IDX = [32, 38, 33, 39, 34, 40, 42]


def query_policy(obs: dict, timeout: float = 60.0) -> dict:
    req_id = uuid.uuid4().hex[:8]
    req_path = f'/tmp/bridge/req/{req_id}.pkl'
    resp_path = f'/tmp/bridge/resp/{req_id}.pkl'
    err_path = f'/tmp/bridge/resp/{req_id}.err'
    tmp_req = f'{req_path}.tmp'
    with open(tmp_req, 'wb') as f:
        pickle.dump(obs, f)
    os.rename(tmp_req, req_path)
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


def build_obs_from_isaac(frame_rgb, joint_pos_np):
    """Build N1.6 obs from Isaac Lab obs.
    
    joint_pos_np: numpy array shape (43,) with all G1 joint positions.
    """
    if frame_rgb is None:
        frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb = np.asarray(frame_rgb, dtype=np.uint8)[None, None]  # (1, 1, H, W, 3)
    
    def _slice(idx):
        return joint_pos_np[idx][None, None].astype(np.float32)
    
    return {
        'video': {'ego_view': rgb},
        'state': {
            'left_leg':   _slice(LEFT_LEG_IDX),    # (1, 1, 6)
            'right_leg':  _slice(RIGHT_LEG_IDX),
            'waist':      _slice(WAIST_IDX),       # (1, 1, 3)
            'left_arm':   _slice(LEFT_ARM_IDX),    # (1, 1, 7)
            'right_arm':  _slice(RIGHT_ARM_IDX),
            'left_hand':  _slice(LEFT_HAND_IDX),
            'right_hand': _slice(RIGHT_HAND_IDX),
        },
        'language': {
            'annotation.human.task_description': [['pick up the apple and place it on the plate']],
        },
    }


def main():
    wait_for_server_ready()
    
    print('[sim] booting Isaac Lab...', flush=True)
    import pinocchio  # noqa: F401
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
    
    # Get robot articulation handle for direct joint state
    robot = env.unwrapped.scene['robot']
    
    def get_joint_pos():
        # Returns numpy (43,) of joint positions
        return robot.data.joint_pos[0].detach().cpu().numpy()
    
    NUM_QUERIES = 8
    STEPS_PER_QUERY = 16  # use most of 30-step chunk
    
    frame = env.render()
    frames = [np.asarray(frame, dtype=np.uint8) if frame is not None else np.zeros((720, 1280, 3), dtype=np.uint8)]
    
    print(f'[sim] starting {NUM_QUERIES} queries x {STEPS_PER_QUERY} steps each ({NUM_QUERIES*STEPS_PER_QUERY} total)', flush=True)
    
    for q in range(NUM_QUERIES):
        # Read REAL state
        jp = get_joint_pos()
        frame = env.render()
        obs_dict = build_obs_from_isaac(frame, jp)
        
        # Query policy
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
            print(f'    sample left_arm[0,:3,:3]:\n{np.asarray(action_chunk["left_arm"])[0,:3,:3]}', flush=True)
        else:
            print(f'[sim] q{q}: {(time.time()-t0)*1000:.0f}ms', flush=True)
        
        # Execute STEPS_PER_QUERY actions from chunk  
        # Map back to Isaac Lab's (1, 28) action space.
        # The Isaac Lab task uses actions for upper body only.
        # Action = [left_arm(7) + right_arm(7) + left_hand(7) + right_hand(7)] = 28
        for s in range(STEPS_PER_QUERY):
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
    
    import imageio
    out_path = '/home/ray/groot_demo/demo_output/g1_groot_n16_realstate.gif'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, frames, fps=15, loop=0)
    print(f'[sim] SAVED: {out_path} ({len(frames)} frames)', flush=True)
    
    with open('/tmp/bridge/STOP', 'w') as f:
        f.write('done')
    
    env.close()
    print('SUCCESS', flush=True)


if __name__ == '__main__':
    main()
