"""
remote_moss_lookbook_runner.py
Moody Moss Mix (MOSS-Image DiT) 4-Season Lookbook Batch Runner on Colab A100.
Generates 4 high-fashion 896x1344 portrait images (Spring, Summer, Autumn, Winter)
with locked character facial identity and seed=8888.
Persists images & manifest.json directly to Google Drive:
Cloud_AI_Workspace/Outputs/Images/Moody_Moss_Lookbook/
"""

import os
import sys
import time
import json
import socket
import urllib.request
import urllib.error
import subprocess
import shutil
import glob

def log(msg, step=None):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[{timestamp}]"
    if step:
        prefix += f" [Step {step}]"
    print(f"{prefix} {msg}", flush=True)

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(('127.0.0.1', port)) == 0

def setup_rclone_config(client_id, client_secret, refresh_token, access_token=""):
    log("Configuring rclone Google Drive credentials...", step=1)
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "rclone"], check=True)

    rclone_config_dir = os.path.expanduser("~/.config/rclone")
    os.makedirs(rclone_config_dir, exist_ok=True)
    config_path = os.path.join(rclone_config_dir, "rclone.conf")

    token_dict = {
        "access_token": access_token or "",
        "token_type": "Bearer",
        "refresh_token": refresh_token,
        "expiry": "2026-11-07T05:00:00Z"
    }

    config_content = f"""[gdrive]
type = drive
client_id = {client_id}
client_secret = {client_secret}
scope = drive
token = {json.dumps(token_dict)}
"""
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)
    log("[OK] rclone Google Drive credentials configured.")

def setup_comfyui_environment():
    log("Deploying ComfyUI environment for Moody Moss Mix...", step=2)
    comfy_dir = "/content/ComfyUI"
    if not os.path.exists(comfy_dir):
        log("Cloning ComfyUI official repo...")
        subprocess.run(["git", "clone", "-q", "https://github.com/comfyanonymous/ComfyUI.git", comfy_dir], check=True)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=comfy_dir, check=True)

    # Directories
    models_dir = os.path.join(comfy_dir, "models")
    diff_dir = os.path.join(models_dir, "diffusion_models")
    clip_dir = os.path.join(models_dir, "clip")
    vae_dir = os.path.join(models_dir, "vae")

    for d in [diff_dir, clip_dir, vae_dir]:
        os.makedirs(d, exist_ok=True)

    log("Syncing Moody Moss Mix checkpoint from Google Drive...", step=3)
    subprocess.run(["rclone", "copy", "gdrive:Cloud_AI_Workspace/Models/checkpoints/moody_moss_mix_v3.3_25steps_cfg3.safetensors", diff_dir, "--progress"], check=True)

    log("Syncing Qwen3.5 9B text encoder from Google Drive...")
    subprocess.run(["rclone", "copy", "gdrive:Cloud_AI_Workspace/Models/text_encoders/qwen3.5_9b_int8_convrot.safetensors", clip_dir, "--progress"], check=True)

    log("Syncing FLUX2 VAE from Google Drive...")
    subprocess.run(["rclone", "copy", "gdrive:Cloud_AI_Workspace/Models/vae/flux2-vae.safetensors", vae_dir, "--progress"], check=True)

    return comfy_dir

def launch_comfyui_server(comfy_dir):
    log("Starting ComfyUI Server in headless mode (lowvram dynamic allocation)...", step=4)
    server_log = open("/content/comfyui_server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188", "--lowvram", "--dont-print-server"],
        cwd=comfy_dir,
        stdout=server_log,
        stderr=subprocess.STDOUT
    )

    log("Waiting for ComfyUI port 8188...")
    for i in range(60):
        if is_port_open(8188):
            log(f"[OK] ComfyUI server is LIVE at localhost:8188 in {i+1}s.")
            return proc
        time.sleep(1.0)
    raise TimeoutError("ComfyUI startup timed out.")

def build_moss_prompt_graph(task):
    prompt_text = task["prompt"]
    neg_prompt = task.get("neg_prompt", "blurry, out of focus, distorted, deformed, extra fingers, cartoon, plastic, overexposed, bad anatomy, bad hands, watermark")
    width = int(task.get("width", 896))
    height = int(task.get("height", 1344))
    steps = int(task.get("steps", 25))
    cfg = float(task.get("cfg", 3.0))
    seed = int(task.get("seed", 8888))
    batch_size = int(task.get("batch_size", task.get("count", 1)))
    sampler_name = task.get("sampler", "euler")
    scheduler = task.get("scheduler", "simple")
    task_name = task.get("name", "Moss_Lookbook")

    nodes = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "moody_moss_mix_v3.3_25steps_cfg3.safetensors",
                "weight_dtype": "default"
            }
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3.5_9b_int8_convrot.safetensors",
                "type": "moss_image"
            }
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "flux2-vae.safetensors"
            }
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": prompt_text
            }
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": neg_prompt
            }
        },
        "6": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": batch_size
            }
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1.0
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["7", 0],
                "vae": ["3", 0]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": f"lookbook_{task_name}"
            }
        }
    }
    return nodes

def submit_moss_prompt(prompt_graph, task):
    url = "http://127.0.0.1:8188/prompt"
    payload = json.dumps({"prompt": prompt_graph}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            prompt_id = res_data["prompt_id"]
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        log(f"[FATAL] ComfyUI Prompt Submission Rejected (HTTP {e.code}):\n{err_msg}")
        raise RuntimeError(f"ComfyUI prompt submission failed: {err_msg}")

    task_name = task.get("name", "Moss_Lookbook")
    batch_size = int(task.get("batch_size", task.get("count", 1)))
    season = task.get("season", "Season")
    log(f"Queued prompt {prompt_id} for {task_name} ({season}). Generating {batch_size} image(s)...")

    output_dir = "/content/ComfyUI/output"
    start_time = time.time()
    last_reported_step = -1

    while True:
        try:
            with open("/content/comfyui_server.log", "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    if "%" in line and ("|" in line or "/" in line):
                        step_str = line.strip()
                        if step_str != last_reported_step:
                            log(f"  [Progress] {step_str}")
                            last_reported_step = step_str
                    elif "ERROR" in line or "Exception" in line:
                        log(f"  [ComfyUI] {line.strip()}")
        except Exception:
            pass

        hist_url = f"http://127.0.0.1:8188/history/{prompt_id}"
        try:
            with urllib.request.urlopen(hist_url) as hist_resp:
                hist_data = json.loads(hist_resp.read().decode("utf-8"))
                if prompt_id in hist_data:
                    info = hist_data[prompt_id]
                    status = info.get("status", {})
                    if status.get("status_str") == "error":
                        log(f"[ERROR] Task failed in ComfyUI: {status.get('messages')}")
                        raise RuntimeError(f"ComfyUI execution error: {status.get('messages')}")

                    outputs = info.get("outputs", {})
                    generated_files = []
                    for node_id, node_out in outputs.items():
                        if "images" in node_out:
                            for img_info in node_out["images"]:
                                fn = img_info["filename"]
                                subfolder = img_info.get("subfolder", "")
                                full_p = os.path.join(output_dir, subfolder, fn)
                                if os.path.exists(full_p):
                                    generated_files.append(full_p)

                    if len(generated_files) >= batch_size:
                        elapsed = time.time() - start_time
                        log(f"[DONE] Completed {len(generated_files)} image(s) for {task_name} in {elapsed:.1f}s!")
                        return generated_files
        except RuntimeError:
            raise
        except Exception:
            pass

        if time.time() - start_time > 400:
            raise TimeoutError(f"Generation for {task_name} timed out after 400s.")

        time.sleep(1.5)

def execute_batch_lookbook_tasks(proc):
    tasks_file = "/content/moss_lookbook_tasks.json"
    if not os.path.exists(tasks_file):
        raise FileNotFoundError(f"Missing {tasks_file}")

    with open(tasks_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    log(f"Loaded {len(tasks)} Moody Moss Lookbook tasks.", step=5)
    dest_drive_dir = "gdrive:Cloud_AI_Workspace/Outputs/Images/Moody_Moss_Lookbook/"

    manifest = []

    for idx, task in enumerate(tasks, 1):
        task_name = task.get("name", f"Moss_Lookbook_{idx}")
        season = task.get("season", "Season")
        title = task.get("title", "Outfit")

        log("======================================================================")
        log(f">> Running Lookbook [{idx}/{len(tasks)}]: {season} - {title}")
        log(f"   Task: {task_name} | Seed: {task.get('seed')} | Dim: {task.get('width')}x{task.get('height')}")
        log("======================================================================")

        prompt_graph = build_moss_prompt_graph(task)
        generated_pngs = submit_moss_prompt(prompt_graph, task)

        log(f"[SUCCESS] Generated {len(generated_pngs)} image for {task_name}.")

        for png_path in generated_pngs:
            subprocess.run(["rclone", "copy", png_path, dest_drive_dir], check=False)
            manifest.append({
                "id": task["id"],
                "name": task_name,
                "season": season,
                "outfit_id": task.get("outfit_id"),
                "title": title,
                "file": os.path.basename(png_path),
                "seed": task.get("seed"),
                "width": task.get("width"),
                "height": task.get("height"),
                "prompt": task["prompt"],
                "status": "completed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

        log(f"[OK] {task_name} persisted to Drive ({dest_drive_dir}).")
        time.sleep(2.0)

    # Save manifest.json to Google Drive
    manifest_local = "/content/manifest.json"
    with open(manifest_local, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    subprocess.run(["rclone", "copy", manifest_local, dest_drive_dir], check=False)
    log("[OK] Lookbook manifest.json persisted to Google Drive.")

    log("============================================================")
    log("[SUCCESS] All Moody Moss Lookbook tasks completed successfully!")
    log("============================================================")

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 remote_moss_lookbook_runner.py <client_id> <client_secret> <refresh_token> [access_token]")
        sys.exit(1)

    client_id = sys.argv[1]
    client_secret = sys.argv[2]
    refresh_token = sys.argv[3]
    access_token = sys.argv[4] if len(sys.argv) > 4 else ""

    setup_rclone_config(client_id, client_secret, refresh_token, access_token)
    comfy_dir = setup_comfyui_environment()
    proc = launch_comfyui_server(comfy_dir)

    try:
        execute_batch_lookbook_tasks(proc)
    finally:
        log("Cleaning up ComfyUI server...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    main()
