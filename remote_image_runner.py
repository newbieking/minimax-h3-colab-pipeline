"""
remote_image_runner.py
High-throughput Image Generation Pipeline for ComfyUI on Colab A100 / GPU.
Supports multi-candidate character batches (4-8 images per prompt),
dynamic LoRA chaining from Google Drive anime_characters/,
and instant syncing of generated high-res images to Google Drive.
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
        "expiry": "2026-10-30T13:00:00Z"
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
    log("Deploying ComfyUI environment...", step=2)
    os.chdir("/content")

    # Install uv for fast dependency management
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    except Exception:
        pass

    comfy_dir = "/content/ComfyUI"
    if not os.path.exists(os.path.join(comfy_dir, "main.py")):
        log("Cloning ComfyUI official repo...")
        if os.path.exists(comfy_dir):
            shutil.rmtree(comfy_dir)
        subprocess.run(["git", "clone", "-q", "https://github.com/comfyanonymous/ComfyUI.git", comfy_dir], check=True)

    os.chdir(comfy_dir)
    subprocess.run(["uv", "pip", "install", "--system", "-q", "-r", "requirements.txt"], check=True)

    # Prepare model directories
    checkpoints_dir = "/content/ComfyUI/models/checkpoints"
    loras_dir = "/content/ComfyUI/models/loras"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(loras_dir, exist_ok=True)

    # 1. Sync anime LoRAs from Google Drive
    log("Syncing LoRA models from Google Drive...", step=3)
    cmd_sync_loras = ["rclone", "copy", "gdrive:MiniMax_Models/loras/", loras_dir, "--progress"]
    subprocess.run(cmd_sync_loras, check=False)

    # 2. Sync checkpoints from Google Drive (including Moody Moss Mix)
    log("Syncing checkpoints from Google Drive...")
    cmd_sync_ckpt = ["rclone", "copy", "gdrive:Cloud_AI_Workspace/Models/checkpoints/", checkpoints_dir, "--progress"]
    subprocess.run(cmd_sync_ckpt, check=False)

    return comfy_dir

def launch_comfyui_server(comfy_dir):
    log("Starting ComfyUI Server in headless background mode...", step=4)
    server_log = open("/content/comfyui_server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188", "--highvram", "--dont-print-server"],
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

def build_comfyui_image_prompt(task):
    """Build ComfyUI API graph with dynamic LoRA chaining and multi-candidate batching."""
    prompt_text = task["prompt"]
    neg_prompt = task.get("negative_prompt", "")
    width = int(task.get("width", 896))
    height = int(task.get("height", 1152))
    steps = int(task.get("steps", 25))
    cfg = float(task.get("cfg", 3.5))
    seed = int(task.get("seed", int(time.time()) % 1000000))
    batch_size = int(task.get("batch_size", 4))
    task_name = task.get("name", "Task_Image")
    base_model_name = task.get("checkpoint", "moody_moss_mix_v3.3_25steps_cfg3.safetensors")
    sampler_name = task.get("sampler", "euler")
    scheduler = task.get("scheduler", "simple")
    loras = task.get("loras", [])

    nodes = {}

    # 1. Load Checkpoint
    nodes["1"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": base_model_name}
    }

    current_model = ["1", 0]
    current_clip = ["1", 1]
    node_id_counter = 10

    # Dynamic LoRA Chaining
    for lora in loras:
        lora_name = lora.get("name")
        lora_strength = float(lora.get("strength", 0.8))
        nid = str(node_id_counter)
        node_id_counter += 1

        nodes[nid] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": current_model,
                "clip": current_clip,
                "lora_name": lora_name,
                "strength_model": lora_strength,
                "strength_clip": lora_strength
            }
        }
        current_model = [nid, 0]
        current_clip = [nid, 1]

    # Positive Prompt
    nodes["2"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": prompt_text,
            "clip": current_clip
        }
    }

    # Negative Prompt
    nodes["3"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": neg_prompt,
            "clip": current_clip
        }
    }

    # Empty Latent Image (Multi-candidate batch)
    nodes["4"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": width,
            "height": height,
            "batch_size": batch_size
        }
    }

    # KSampler
    nodes["5"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": 1.0,
            "model": current_model,
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0]
        }
    }

    # VAE Decode
    nodes["6"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["5", 0],
            "vae": ["1", 2]
        }
    }

    # Save Image
    nodes["7"] = {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": f"character_{task_name}",
            "images": ["6", 0]
        }
    }

    return nodes

def stream_comfyui_logs(server_log_path, last_pos):
    if not os.path.exists(server_log_path):
        return last_pos
    try:
        with open(server_log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(last_pos)
            new_lines = f.readlines()
            last_pos = f.tell()
            for line in new_lines:
                line_clean = line.rstrip()
                if line_clean and not any(k in line_clean for k in ["GET /history", "GET /queue", "GET /prompt"]):
                    log(f"  [ComfyUI] {line_clean}")
    except Exception:
        pass
    return last_pos

def submit_image_prompt(prompt_dict, task, timeout=400):
    client_id = f"img_runner_{int(time.time())}"
    post_data = json.dumps({"prompt": prompt_dict, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=post_data, headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(req, timeout=15) as resp:
        resp_data = json.loads(resp.read().decode("utf-8"))

    prompt_id = resp_data.get("prompt_id")
    log(f"Queued prompt {prompt_id}. Generating {task.get('batch_size', 4)} image candidates...")

    start_t = time.time()
    output_dir = "/content/ComfyUI/output"
    os.makedirs(output_dir, exist_ok=True)
    server_log_path = "/content/comfyui_server.log"
    log_pos = 0

    while time.time() - start_t < timeout:
        log_pos = stream_comfyui_logs(server_log_path, log_pos)
        hist_url = f"http://127.0.0.1:8188/history/{prompt_id}"
        try:
            with urllib.request.urlopen(hist_url, timeout=5) as h_resp:
                hist_data = json.loads(h_resp.read().decode("utf-8"))
            if prompt_id in hist_data:
                item = hist_data[prompt_id]
                status = item.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"Prompt execution error: {status.get('messages')}")
                if status.get("completed", False):
                    log(f"[OK] Prompt {prompt_id} execution completed.")
                    break
        except Exception:
            pass
        time.sleep(2.0)
    else:
        raise TimeoutError(f"Task {task.get('name')} timed out.")

    # Find generated PNG images
    task_name = task.get("name", "")
    all_pngs = []
    for f in os.listdir(output_dir):
        if f.endswith(".png") and task_name in f:
            all_pngs.append(os.path.join(output_dir, f))
    return all_pngs

def execute_batch_image_tasks(proc):
    log("Starting Batch Character Image Generation...", step=5)
    tasks_path = "/content/image_tasks.json"
    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    log(f"Loaded {len(tasks)} image generation tasks.")

    for idx, task in enumerate(tasks, start=1):
        task_name = task.get("name")
        batch_size = task.get("batch_size", 4)
        log("----------------------------------------------------------------------")
        log(f">> Running Task [{idx}/{len(tasks)}]: {task_name} (Batch Size: {batch_size} Images)")
        log("----------------------------------------------------------------------")

        prompt_graph = build_comfyui_image_prompt(task)
        generated_pngs = submit_image_prompt(prompt_graph, task)

        log(f"[SUCCESS] Generated {len(generated_pngs)} candidates for {task_name}.")

        # Upload each batch to Google Drive: MiniMax_H3_Outputs/Generated_Images/
        log(f"Uploading batch to Google Drive (gdrive:Cloud_AI_Workspace/Outputs/Images/Character_LookDev/{task_name}/)...")
        dest_drive = f"gdrive:Cloud_AI_Workspace/Outputs/Images/Character_LookDev/{task_name}/"
        for png_path in generated_pngs:
            subprocess.run(["rclone", "copy", png_path, dest_drive], check=False)

        log(f"[OK] {task_name} image batch persisted to Google Drive.")

    log("All image tasks completed successfully!")

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 remote_image_runner.py <client_id> <client_secret> <refresh_token> [access_token]")
        sys.exit(1)

    client_id = sys.argv[1]
    client_secret = sys.argv[2]
    refresh_token = sys.argv[3]
    access_token = sys.argv[4] if len(sys.argv) > 4 else ""

    setup_rclone_config(client_id, client_secret, refresh_token, access_token)
    comfy_dir = setup_comfyui_environment()
    proc = launch_comfyui_server(comfy_dir)

    try:
        execute_batch_image_tasks(proc)
    finally:
        log("Cleaning up ComfyUI server...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    main()
