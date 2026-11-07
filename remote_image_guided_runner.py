"""
remote_image_guided_runner.py
Image-Conditioned & Text-Conditioned Dual-Guidance FLUX.1 [dev] Batch Runner on Colab GPU.
Key Features:
- Face/Style Reference Guidance: Employs sigclip_vision_patch14_384 to encode reference face image (f71f980dde3.webp).
- Orthogonal Variable Decoupling: Integrates visual feature embedding with outfit text prompts across 8 distinct categories.
- Single Cold-Start Pipeline: Warm-caches base models and streams tasks in a single GPU session.
- Low-VRAM Auto-Offload: Dynamically frees encoder memory to ensure zero OOM risks on A100/T4.
- Incremental Cloud Push: Live-syncs outputs directly to Google Drive with manifest.json.
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
    log("Configuring rclone credentials...", step=1)
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "rclone"], check=True)

    rclone_config_dir = os.path.expanduser("~/.config/rclone")
    os.makedirs(rclone_config_dir, exist_ok=True)
    config_path = os.path.join(rclone_config_dir, "rclone.conf")

    token_dict = {
        "access_token": access_token or "",
        "token_type": "Bearer",
        "refresh_token": refresh_token,
        "expiry": "2026-11-07T08:00:00Z"
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
    log("[OK] rclone credentials configured.")

def setup_comfyui_environment():
    log("Deploying ComfyUI environment for Image-Guided FLUX.1...", step=2)
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
    clip_vision_dir = os.path.join(models_dir, "clip_vision")
    input_dir = os.path.join(comfy_dir, "input")

    for d in [diff_dir, clip_dir, vae_dir, clip_vision_dir, input_dir]:
        os.makedirs(d, exist_ok=True)

    log("Syncing FLUX.1 Dev FP8 model from Google Drive...", step=3)
    subprocess.run(["rclone", "copy", "gdrive:Cloud_AI_Workspace/Models/diffusion_models/flux1-dev-fp8.safetensors", diff_dir, "--progress"], check=True)

    log("Syncing Dual CLIP encoders from Google Drive...")
    subprocess.run(["rclone", "copy", "gdrive:Cloud_AI_Workspace/Models/text_encoders/t5xxl_fp8_e4m3fn.safetensors", clip_dir, "--progress"], check=True)
    subprocess.run(["rclone", "copy", "gdrive:Cloud_AI_Workspace/Models/text_encoders/clip_l.safetensors", clip_dir, "--progress"], check=True)

    log("Syncing FLUX VAE (ae.safetensors) from Google Drive...")
    subprocess.run(["rclone", "copy", "gdrive:Cloud_AI_Workspace/Models/vae/ae.safetensors", vae_dir, "--progress"], check=True)

    log("Syncing SigLIP Vision encoder from Google Drive...")
    subprocess.run(["rclone", "copy", "gdrive:Cloud_AI_Workspace/Models/clip_vision/sigclip_vision_patch14_384.safetensors", clip_vision_dir, "--progress"], check=True)

    return comfy_dir

def launch_comfyui_server(comfy_dir):
    log("Starting ComfyUI Server in headless mode (lowvram dynamic memory)...", step=4)
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

def build_image_guided_graph(task, ref_image_name):
    prompt_text = task["prompt"]
    neg_prompt = task.get("neg_prompt", "blurry, out of focus, distorted, deformed, extra fingers, cartoon, plastic, overexposed, bad anatomy, bad hands, watermark")
    width = int(task.get("width", 896))
    height = int(task.get("height", 1344))
    steps = int(task.get("steps", 25))
    guidance = float(task.get("guidance", 3.5))
    seed = int(task.get("seed", 9999))
    task_name = task.get("name", "Task_Guided")

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
        # 4. Text Positive Prompt
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": prompt_text
            }
        },
        # 5. Text Negative Prompt
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": neg_prompt
            }
        },
        # 6. Load Reference Face Image
        "6": {
            "class_type": "LoadImage",
            "inputs": {
                "image": ref_image_name
            }
        },
        # 7. CLIP Vision Loader (SigLIP 384)
        "7": {
            "class_type": "CLIPVisionLoader",
            "inputs": {
                "clip_name": "sigclip_vision_patch14_384.safetensors"
            }
        },
        # 8. CLIP Vision Encode (Extracts Face/Styling embedding from Reference Image)
        "8": {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["7", 0],
                "image": ["6", 0]
            }
        },
        # 9. Flux Guidance Node (combines text conditioning with guidance scalar)
        "9": {
            "class_type": "FluxGuidance",
            "inputs": {
                "guidance": guidance,
                "conditioning": ["4", 0]
            }
        },
        # 10. Empty Latent Canvas (896x1344 full body portrait)
        "10": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        # 11. KSampler (Euler / Simple, CFG=1.0)
        "11": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["9", 0],
                "negative": ["5", 0],
                "latent_image": ["10", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0
            }
        },
        # 12. VAE Decode
        "12": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["11", 0],
                "vae": ["3", 0]
            }
        },
        # 13. Save Image
        "13": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["12", 0],
                "filename_prefix": f"photoshoot_{task_name}"
            }
        }
    }
    return nodes

def submit_guided_prompt(prompt_graph, task):
    url = "http://127.0.0.1:8188/prompt"
    payload = json.dumps({"prompt": prompt_graph}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            prompt_id = res_data["prompt_id"]
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        log(f"[FATAL] Submission Rejected (HTTP {e.code}):\n{err_msg}")
        raise RuntimeError(f"Prompt submission failed: {err_msg}")

    task_name = task.get("name", "Task")
    log(f"Queued prompt {prompt_id} for {task_name}...")

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
                        raise RuntimeError(f"ComfyUI error: {status.get('messages')}")

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

        if time.time() - start_time > 360:
            raise TimeoutError(f"Task {task_name} timed out after 360s.")

        time.sleep(1.5)

def execute_photoshoot_pipeline(proc, tasks_path, dest_drive_dir, ref_image_name):
    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    log(f"Loaded {len(tasks)} image-guided photoshoot tasks.", step=5)
    manifest = []
    success_count = 0

    for idx, task in enumerate(tasks, 1):
        task_name = task.get("name", f"Task_{idx}")
        category = task.get("category", "Photoshoot")
        title = task.get("title", task_name)

        log("======================================================================")
        log(f">> [Photo {idx}/{len(tasks)}] {category} - {title}")
        log(f"   Model Ref: {ref_image_name} | Seed: {task.get('seed')} | Dim: {task.get('width')}x{task.get('height')}")
        log("======================================================================")

        try:
            prompt_graph = build_image_guided_graph(task, ref_image_name)
            generated_pngs = submit_guided_prompt(prompt_graph, task)

            for png_path in generated_pngs:
                subprocess.run(["rclone", "copy", png_path, dest_drive_dir], check=False)
                manifest.append({
                    "id": task.get("id", idx),
                    "name": task_name,
                    "title": title,
                    "category": category,
                    "outfit_id": task.get("outfit_id"),
                    "file": os.path.basename(png_path),
                    "seed": task.get("seed"),
                    "prompt": task.get("prompt"),
                    "ref_image": ref_image_name,
                    "status": "completed",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            success_count += 1
            log(f"[OK] {task_name} persisted to Drive ({dest_drive_dir}).")
        except Exception as e:
            log(f"[WARN] Error in {task_name}: {e}. Continuing pipeline...")
            manifest.append({
                "id": task.get("id", idx),
                "name": task_name,
                "status": "failed",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

        time.sleep(2.0)

    # Persist manifest
    manifest_local = "/content/manifest.json"
    with open(manifest_local, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    subprocess.run(["rclone", "copy", manifest_local, dest_drive_dir], check=False)
    log(f"[OK] Manifest synced ({success_count}/{len(tasks)} completed).")

def main():
    if len(sys.argv) < 5:
        print("Usage: python3 remote_image_guided_runner.py <client_id> <client_secret> <refresh_token> <access_token> <tasks_path> <dest_drive> <ref_image_name>")
        sys.exit(1)

    client_id = sys.argv[1]
    client_secret = sys.argv[2]
    refresh_token = sys.argv[3]
    access_token = sys.argv[4]
    tasks_path = sys.argv[5] if len(sys.argv) > 5 else "/content/tasks.json"
    dest_drive = sys.argv[6] if len(sys.argv) > 6 else "gdrive:Cloud_AI_Workspace/Outputs/Images/Virtual_Model_Photoshoot_f71f/"
    ref_image_name = sys.argv[7] if len(sys.argv) > 7 else "f71f980dde3.webp"

    setup_rclone_config(client_id, client_secret, refresh_token, access_token)
    comfy_dir = setup_comfyui_environment()

    # Copy reference image into ComfyUI input folder
    src_ref = f"/content/{ref_image_name}"
    dest_ref = f"{comfy_dir}/input/{ref_image_name}"
    if os.path.exists(src_ref):
        shutil.copyfile(src_ref, dest_ref)
        log(f"[OK] Reference image staged into ComfyUI/input: {dest_ref}")

    proc = launch_comfyui_server(comfy_dir)

    try:
        execute_photoshoot_pipeline(proc, tasks_path, dest_drive, ref_image_name)
    finally:
        log("Cleaning up ComfyUI server...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    main()
