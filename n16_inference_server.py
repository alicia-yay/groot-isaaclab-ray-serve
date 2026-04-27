"""
N1.6 G1 inference server — runs in groot-n16 conda env.
Watches /tmp/bridge/req/*.pkl, writes /tmp/bridge/resp/*.pkl.
Same-worker, file-based RPC. Simple and avoids env conflicts.
"""
import sys, os, time, pickle, glob
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/ray/Isaac-GR00T-N16')

# Patches
import transformers.image_utils
from transformers.models.auto.auto_factory import _BaseAutoModelClass
_orig = _BaseAutoModelClass.from_config.__func__
def _p(cls, config, **kwargs):
    if hasattr(config, 'text_config'):
        config.text_config._attn_implementation = 'flash_attention_2'
    config._attn_implementation = 'flash_attention_2'
    if 'attn_implementation' not in kwargs:
        kwargs['attn_implementation'] = 'flash_attention_2'
    return _orig(cls, config, **kwargs)
_BaseAutoModelClass.from_config = classmethod(_p)

from huggingface_hub import login
tok = os.environ.get('HF_TOKEN')
if tok:
    login(token=tok, add_to_git_credential=False)

import gr00t.model
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag
import numpy as np

print('[server] loading policy (~60s)...', flush=True)
t0 = time.time()
policy = Gr00tPolicy(
    embodiment_tag=EmbodimentTag.UNITREE_G1,
    model_path='nvidia/GR00T-N1.6-G1-PnPAppleToPlate',
    device='cuda:0',
)
print(f'[server] ready in {time.time()-t0:.1f}s', flush=True)

# Signal ready
os.makedirs('/tmp/bridge/req', exist_ok=True)
os.makedirs('/tmp/bridge/resp', exist_ok=True)
with open('/tmp/bridge/READY', 'w') as f:
    f.write(str(time.time()))

print('[server] watching /tmp/bridge/req/...', flush=True)

while True:
    if os.path.exists('/tmp/bridge/STOP'):
        print('[server] STOP signal received', flush=True)
        break
    
    reqs = sorted(glob.glob('/tmp/bridge/req/*.pkl'))
    if not reqs:
        time.sleep(0.05)
        continue
    
    req_path = reqs[0]
    req_id = os.path.basename(req_path).replace('.pkl', '')
    try:
        with open(req_path, 'rb') as f:
            obs = pickle.load(f)
        os.remove(req_path)
        
        t0 = time.time()
        action, info = policy.get_action(obs)
        latency_ms = (time.time() - t0) * 1000
        
        resp = {'action': action, 'latency_ms': latency_ms, 'info': info}
        # Atomic write: tmp file then rename
        tmp_path = f'/tmp/bridge/resp/{req_id}.pkl.tmp'
        final_path = f'/tmp/bridge/resp/{req_id}.pkl'
        with open(tmp_path, 'wb') as f:
            pickle.dump(resp, f)
        os.rename(tmp_path, final_path)
        print(f'[server] {req_id}: {latency_ms:.0f}ms', flush=True)
    except Exception as e:
        import traceback
        err_path = f'/tmp/bridge/resp/{req_id}.err'
        with open(err_path, 'w') as f:
            f.write(traceback.format_exc())
        print(f'[server] {req_id} ERROR: {e}', flush=True)
        if os.path.exists(req_path):
            os.remove(req_path)

print('[server] done', flush=True)
