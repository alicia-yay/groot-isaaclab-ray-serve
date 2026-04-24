"""
Push HF_TOKEN to every Ray worker node so the `hf download` and
`from_pretrained` calls can fetch gated models.

GR00T N1.6 base is open; the G1 fine-tune is also open. But flash-attn's
source download + some dependent VLMs may want auth. Setting the token
everywhere avoids surprises.

Usage:
    export HF_TOKEN=hf_your_token_here
    python set_token.py
"""
import os
import ray


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set. `export HF_TOKEN=hf_...` first.")

    ray.init(address="auto", ignore_reinit_error=True)

    @ray.remote(num_gpus=1)
    def write_token(tok: str):
        import os
        home = os.path.expanduser("~")
        os.makedirs(f"{home}/.cache/huggingface", exist_ok=True)
        with open(f"{home}/.cache/huggingface/token", "w") as f:
            f.write(tok)
        # Also set env for this worker's shell profile.
        with open(f"{home}/.bashrc", "a") as f:
            f.write(f"\nexport HF_TOKEN={tok}\n")
        return f"ok: {os.uname().nodename}"

    num_nodes = int(ray.cluster_resources().get("GPU", 0))
    results = ray.get([write_token.remote(token) for _ in range(num_nodes)])
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
