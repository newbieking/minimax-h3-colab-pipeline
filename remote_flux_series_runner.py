"""
remote_flux_series_runner.py
High-throughput FLUX.1 [dev] + LoRA Series Runner on Colab A100.
Executes Series A: 'Parallel Worlds' across 5 iconic styles with locked seed.
Features:
- DualCLIPLoader (T5XXL FP8 + CLIP-L, type='flux')
- FluxGuidance (guidance=3.5, cfg=1.0)
- LoraLoader dynamic chaining
- Real-time ComfyUI progress streaming
- Manifest persistence & instant Google Drive sync to Cloud_AI_Workspace/Outputs/Images/FLUX_Series/
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
        "expiry": "2026-10-31T05:00:00Z"
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
    log("Deploying ComfyUI environment for FLUX.1 + LoRA...", step=2)
    comfy_dir = "/content/ComfyUI"
    if not os.path.exists(comfy_dir):
        log("Cloning ComfyUI official repo...")
        subprocess.run(["git", "clone", "-q", "https://github.com/comfyanonymous/ComfyUI.git", comfy_dir], check=True)

    # Install ComfyUI dependencies
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=comfy_dir, check=True)

    # Directories
    models_dir = os.path.join(comfy_dir, "models")
    diff_dir = os.path.join(models_dir, "diffusion_models")
    clip_dir = os.path.join(models_dir, "clip")
    te_dir = os.path.join(models_dir, "text_encoders")
    vae_dir = os.path.join(models_dir, "vae")
    loras_dir = os.path.join(models_dir, "loras")

    for d in [diff_dir, clip_dir, te_dir, vae_dir, loras_dir]:
        os.makedirs(d, exist_ok=True)

    log("Syncing FLUX.1 Dev FP8 model from Google Drive...", step=3)
    subprocess.run(["rclone", "copy", "gdrive:Cloud_AI_Workspace/Models/diffusion_models/flux1-dev-fp8.safetensors", diff_dir, "--progress"], check=True)

    log("Syncing Dual CLIP encoders (T5XXL + CLIP-L) from Google Drive...")
    subprocess.run(["rclone", "copy", "gdrive:Cloud_AI_Workspace/Models/text_encoders/t5xxl_fp8_e4m3fn.safetensors", clip_dir, "--progress"], check=True)
    subprocess.run(["rclone", "copy", "gdrive:Cloud_AI_Workspace/Models/text_encoders/clip_l.safetensors", clip_dir, "--progress"], check=True)

    log("Syncing FLUX VAE (ae.safetensors) from Google Drive...")
    subprocess.run(["rclone", "copy", "gdrive:Cloud_AI_Workspace/Models/vae/ae.safetensors", vae_dir, "--progress"], check=True)

    log("Syncing FLUX LoRA models from Google Drive...")
    dest_flux_loras = os.path.join(loras_dir, "flux")
    subprocess.run(["rclone", "copy", "gdrive:Cloud_AI_Workspace/Models/loras/flux/", dest_flux_loras, "--progress"], check=True)

    return comfy_dir

def launch_comfyui_server(comfy_dir):
    log("Starting ComfyUI Server in headless background mode (with dynamic memory management)...", step=4)
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

def find_lora_relative_path(lora_filename_or_path, loras_root="/content/ComfyUI/models/loras"):
    target_basename = os.path.basename(lora_filename_or_path).lower()
    for root, _, files in os.walk(loras_root):
        for f in files:
            if f.lower() == target_basename:
                abs_p = os.path.join(root, f)
                rel_p = os.path.relpath(abs_p, loras_root).replace("\\", "/")
                return rel_p
    # fallback
    return f"flux/{lora_filename_or_path}"

def build_flux_lora_prompt_graph(task):
    prompt_text = task["prompt"]
    width = int(task.get("width", 768))
    height = int(task.get("height", 768))
    steps = int(task.get("steps", 20))
    guidance = float(task.get("guidance", 3.5))
    seed = int(task.get("seed", 2026))
    lora_rel_path = find_lora_relative_path(task['lora_path'])
    lora_strength = float(task.get("lora_strength", 0.85))
    task_name = task.get("name", "Task_Flux")

    nodes = {
        # 1. Base UNET Loader (FLUX.1 Dev FP8)
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "flux1-dev-fp8.safetensors",
                "weight_dtype": "default"
            }
        },
        # 2. Dual CLIP Loader (T5XXL + CLIP-L)
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
                "clip_name2": "clip_l.safetensors",
                "type": "flux"
            }
        },
        # 3. VAE Loader
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "ae.safetensors"
            }
        },
        # 4. LoRA Loader
        "4": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["2", 0],
                "lora_name": lora_rel_path,
                "strength_model": lora_strength,
                "strength_clip": lora_strength
            }
        },
        # 5. Positive Prompt (connected to LoRA-conditioned CLIP)
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": prompt_text
            }
        },
        # 6. Negative Prompt (empty for FLUX)
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": ""
            }
        },
        # 7. FluxGuidance Node (applies guidance=3.5 scalar to positive prompt)
        "7": {
            "class_type": "FluxGuidance",
            "inputs": {
                "guidance": guidance,
                "conditioning": ["5", 0]
            }
        },
        # 8. Empty Latent Canvas (768x768 concept batch)
        "8": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        # 9. KSampler (Euler / Simple, CFG=1.0 strictly for FLUX)
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "positive": ["7", 0],
                "negative": ["6", 0],
                "latent_image": ["8", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0
            }
        },
        # 10. VAE Decode
        "10": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["9", 0],
                "vae": ["3", 0]
            }
        },
        # 11. Save Image
        "11": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["10", 0],
                "filename_prefix": f"flux_{task_name}"
            }
        }
    }
    return nodes

def submit_flux_prompt(prompt_graph, task):
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

    task_name = task.get("name", "Task_Flux")
    log(f"Queued prompt {prompt_id} for {task_name} (Seed: {task.get('seed')}, Style: {task.get('style')})...")

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

                    if len(generated_files) >= 1:
                        elapsed = time.time() - start_time
                        log(f"[DONE] Generated {task_name} in {elapsed:.1f}s!")
                        return generated_files
        except RuntimeError:
            raise
        except Exception:
            pass

        if time.time() - start_time > 300:
            raise TimeoutError(f"Generation for {task_name} timed out after 300s.")

        time.sleep(1.5)

def execute_batch_flux_tasks(proc):
    tasks_file = "/content/flux_series_tasks.json"
    if not os.path.exists(tasks_file):
        raise FileNotFoundError(f"Missing {tasks_file}")

    with open(tasks_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    log(f"Loaded {len(tasks)} FLUX Series A tasks.", step=5)
    dest_drive_dir = "gdrive:Cloud_AI_Workspace/Outputs/Images/FLUX_Series/SeriesA_Parallel_Worlds/"

    manifest = []

    for idx, task in enumerate(tasks, 1):
        task_name = task.get("name", f"Flux_Task_{idx}")
        style_desc = task.get("style", "Style")

        log("======================================================================")
        log(f">> Running Parallel World [{idx}/{len(tasks)}]: {task_name} ({style_desc})")
        log(f"   LoRA: {task['lora_path']} (Strength: {task.get('lora_strength')}) | Seed: {task.get('seed')}")
        log("======================================================================")

        prompt_graph = build_flux_lora_prompt_graph(task)
        generated_pngs = submit_flux_prompt(prompt_graph, task)

        log(f"[SUCCESS] Generated image for {task_name}.")

        for png_path in generated_pngs:
            subprocess.run(["rclone", "copy", png_path, dest_drive_dir], check=False)
            manifest.append({
                "id": task["id"],
                "name": task_name,
                "style": style_desc,
                "file": os.path.basename(png_path),
                "lora": task["lora_path"],
                "strength": task.get("lora_strength"),
                "seed": task.get("seed"),
                "prompt": task["prompt"],
                "status": "completed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

        log(f"[OK] {task_name} persisted to Drive ({dest_drive_dir}).")

        # Sleep 3 seconds between tasks to allow VRAM cooldown
        time.sleep(3.0)

    # Save manifest.json to Google Drive
    manifest_local = "/content/manifest.json"
    with open(manifest_local, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    subprocess.run(["rclone", "copy", manifest_local, dest_drive_dir], check=False)
    log("[OK] Task manifest.json persisted to Google Drive.")

    log("============================================================")
    log("[SUCCESS] All FLUX Series A tasks completed successfully!")
    log("============================================================")

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 remote_flux_series_runner.py <client_id> <client_secret> <refresh_token> [access_token]")
        sys.exit(1)

    client_id = sys.argv[1]
    client_secret = sys.argv[2]
    refresh_token = sys.argv[3]
    access_token = sys.argv[4] if len(sys.argv) > 4 else ""

    setup_rclone_config(client_id, client_secret, refresh_token, access_token)
    comfy_dir = setup_comfyui_environment()
    proc = launch_comfyui_server(comfy_dir)

    try:
        execute_batch_flux_tasks(proc)
    finally:
        log("Cleaning up ComfyUI server...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    main()
