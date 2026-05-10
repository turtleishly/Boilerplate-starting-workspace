
# USE `modal run start_jupyter.py` to run
# Note, the printed link is broken at the linebreak

import modal
import secrets
from pathlib import Path

# 1. Define your "Dream Environment"
requirements_path = Path(__file__).with_name("requirements.txt")

# Looks for .env starting from this script's directory
dotenv_secret = modal.Secret.from_dotenv(__file__)

# Persistent cache for Hugging Face model downloads
models_vol = modal.Volume.from_name("models-cache-vol", create_if_missing=True)


my_image = (
    modal.Image.debian_slim()
    .pip_install(
        "uv",
        "jupyterlab",
        "torch",
        "transformer_lens",
        "sae_lens", # Added since you use it in imports
        "accelerate",
        "huggingface_hub",
        "matplotlib",
        "circuitsvis",
        "plotly",
        "jaxtyping",
        "peft"
    )
    # removed .run_commands(...) to avoid local-path installs
)

app = modal.App("custom-mech-interp", image=my_image)

@app.local_entrypoint()
def main():
    print("🚀 Spinning up your custom A100 environment...")

    token = secrets.token_urlsafe(16)

    # 2. Launch the sandbox
    # FIX: We explicitly pass 'app=app' so it knows which session to attach to
    sandbox = modal.Sandbox.create(
        "jupyter", "lab",
        "--no-browser",
        "--ip=0.0.0.0",
        "--port=8888",
        "--allow-root",
        f"--ServerApp.token={token}",
        "--ServerApp.disable_check_xsrf=True",
        "--ServerApp.allow_origin='*'",
        image=my_image,
        gpu="A100-80GB",  # CHANGE THIS TO PREFERRED to "T4" if you want to save credits
        encrypted_ports=[8888],
        timeout=3600, # Safety net: auto-shutdown after 1 hour
        app=app,
        volumes={
            "/root/cache": models_vol,
        },
        secrets=[
            dotenv_secret,
            modal.Secret.from_dict(
                {
                    "HF_HOME": "/root/cache",
                    "TRANSFORMERS_CACHE": "/root/cache/transformers",
                    "HF_HUB_CACHE": "/root/cache/hub",
                }
            ),
        ],  # injects .env keys as env vars
    )

    # 3. Get the secure URL
    tunnel_url = sandbox.tunnels()[8888].url
    print(f"\n✅ SUCCESS! Your GPU Server is ready.")
    print(f"🔗 COPY THIS URL into VS Code 'Existing Jupyter Server':")
    print(f"{tunnel_url}/lab?token={token}\n")
    
    # 4. Keep it running until you stop it
    try:
        sandbox.wait()
    except KeyboardInterrupt:
        print("Stopping server...")
        sandbox.terminate()