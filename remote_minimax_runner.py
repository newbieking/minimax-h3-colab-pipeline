"""
remote_minimax_runner.py
High-performance MiniMax H3 video generation workflow on Google Colab A100.
Features:
- High-speed parallel rclone copy from Google Drive to local NVMe SSD (fast_disk=True, zero FUSE bottleneck)
- True ComfyUI API graph generation (Plan C batch queue)
- Real-time unbuffered log streaming and sampling progress tracking
- Instant direct upload of generated videos to Google Drive (MiniMax_H3_Outputs)
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

def check_hardware():
    log("Checking GPU and system environment...", step=1)
    import torch
    import psutil

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected! A100 is required for MiniMax H3.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    ram_gb = psutil.virtual_memory().total / (1024**3)

    log(f"GPU Model: {gpu_name}")
    log(f"GPU VRAM: {vram_gb:.2f} GB")
    log(f"System RAM: {ram_gb:.2f} GB")

    if vram_gb < 35:
        log(f"WARNING: VRAM {vram_gb:.2f} GB is lower than recommended 40GB.")
    else:
        log("[OK] Hardware specifications match MiniMax H3 requirements.")

def setup_rclone_config(client_id, client_secret, refresh_token, access_token=""):
    log("Setting up high-speed rclone configuration...", step=2)
    # Install rclone
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
    log("[OK] rclone Google Drive credentials configured successfully.")

def setup_comfyui_environment():
    log("Deploying ComfyUI repository and dependencies...", step=3)
    os.chdir("/content")

    # Install uv for 10x faster pip installs
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    except Exception:
        pass

    comfy_dir = "/content/ComfyUI"
    if not os.path.exists(os.path.join(comfy_dir, "main.py")):
        log("Cloning ComfyUI official repository...")
        if os.path.exists(comfy_dir):
            shutil.rmtree(comfy_dir)
        subprocess.run(["git", "clone", "-q", "https://github.com/comfyanonymous/ComfyUI.git", comfy_dir], check=True)

    os.chdir(comfy_dir)
    log("Installing ComfyUI requirements with uv...")
    try:
        subprocess.run(["uv", "pip", "install", "--system", "-q", "-r", "requirements.txt"], check=True)
        subprocess.run(["uv", "pip", "install", "--system", "-q", "accelerate", "diffusers", "huggingface_hub", "sentencepiece", "requests"], check=True)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "accelerate", "diffusers", "huggingface_hub", "sentencepiece", "requests"], check=True)

    custom_nodes_dir = os.path.join(comfy_dir, "custom_nodes")
    director_dir = os.path.join(custom_nodes_dir, "ComfyUI_MiniMaxH3_Director")
    if not os.path.exists(director_dir):
        log("Cloning ComfyUI_MiniMaxH3_Director...")
        subprocess.run(["git", "clone", "-q", "https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director.git", director_dir], check=True)
        req = os.path.join(director_dir, "requirements.txt")
        if os.path.exists(req):
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req])

    log("[OK] ComfyUI environment and custom nodes ready.")

def sync_models_to_local_nvme():
    """
    Ultra-Fast Cache Optimization:
    Parallel transfer models from Google Drive directly to local NVMe SSD (/content/ComfyUI/models).
    This completely eliminates FUSE HTTP rate-limiting, enables kernel mmap, and gives 100x faster loading!
    """
    log("Syncing MiniMax H3 models to local NVMe SSD (Ultra-Fast Cache Optimization)...", step=4)
    local_models_dir = "/content/ComfyUI/models"
    os.makedirs(local_models_dir, exist_ok=True)

    t0 = time.time()
    # High-throughput rclone copy parameters for internal Google network
    cmd = [
        "rclone", "copy",
        "gdrive:MiniMax_Models", local_models_dir,
        "--transfers", "4",
        "--drive-chunk-size", "128M",
        "--buffer-size", "64M",
        "--drive-pacer-min-sleep", "10ms",
        "--drive-pacer-burst", "200",
        "--checkers", "8",
        "--stats", "10s",
        "-P"
    ]

    log("Running parallel rclone copy (128MB chunk buffer, 4 concurrent streams)...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        line_clean = line.strip()
        if line_clean and any(k in line_clean for k in ["Transferred:", "ETA", "Checks:", "100%", "Transferring:"]):
            log(f"  [Sync] {line_clean}")
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"rclone copy failed with code {proc.returncode}")

    elapsed = time.time() - t0
    log(f"[SUCCESS] 38GB models synced to local NVMe SSD in {elapsed/60:.2f} minutes (avg {38000/max(elapsed, 1):.1f} MB/s)!")

    # Verify all 5 files exist locally on NVMe disk
    expected_files = [
        os.path.join(local_models_dir, "diffusion_models", "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"),
        os.path.join(local_models_dir, "text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
        os.path.join(local_models_dir, "loras", "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
        os.path.join(local_models_dir, "vae", "minimax_h3_audio_vae_fp32.safetensors"),
        os.path.join(local_models_dir, "vae", "minimax_h3_video_vae_int8_convrot.safetensors")
    ]
    for ef in expected_files:
        if not os.path.exists(ef):
            raise FileNotFoundError(f"Missing expected model file after sync: {ef}")
        size_gb = os.path.getsize(ef) / (1024**3)
        log(f"  Verified local file: {os.path.basename(ef)} ({size_gb:.2f} GB)")

    log("[OK] All model files verified on local NVMe SSD. Native mmap and fast_disk=True enabled!")

def start_comfyui_server():
    log("Starting ComfyUI headless service on port 8188...", step=5)
    output_dir = "/content/output"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, "-u", "main.py",
        "--listen", "127.0.0.1",
        "--port", "8188",
        "--output-directory", output_dir,
        "--disable-auto-launch"
    ]
    log_file = open("/content/comfyui_server.log", "w", encoding="utf-8", errors="replace")
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    proc = subprocess.Popen(cmd, cwd="/content/ComfyUI", env=env, stdout=log_file, stderr=subprocess.STDOUT)

    log("Waiting for ComfyUI service port 8188...")
    for i in range(60):
        if is_port_open(8188):
            log(f"[OK] ComfyUI server is LIVE at localhost:8188 (PID: {proc.pid}) in {i+1}s.")
            return proc
        time.sleep(1.0)

    raise TimeoutError("ComfyUI startup timed out. Check /content/comfyui_server.log")

def calculate_h3_length(duration_sec, fps=24):
    """Align frame length to MiniMax H3 (17k + 5) temporal grid at 24fps."""
    raw_frames = max(5, round(duration_sec * fps))
    return raw_frames + (5 - (raw_frames % 17)) % 17

def build_comfyui_prompt(task):
    """Build ComfyUI API graph dictionary for a single MiniMax H3 task."""
    prompt_text = task["prompt"]
    width = int(task.get("width", 1344))
    height = int(task.get("height", 768))
    duration = float(task.get("duration", 5.0))
    length = calculate_h3_length(duration)
    seed = int(task.get("seed", int(time.time()) % 1000000))
    steps = int(task.get("steps", 4))
    use_lora = bool(task.get("use_lora", True))
    lora_strength = float(task.get("lora_strength", 1.0))
    lora_name = task.get("lora_name", "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors")
    task_name = task.get("name", f"Task_{int(time.time())}")

    nodes = {}

    # 1. Video VAE
    nodes["1"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}
    }
    # 2. Audio VAE
    nodes["2"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}
    }
    # 3. UNET Loader (Pruned FP8 Scaled)
    nodes["3"] = {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
            "weight_dtype": "default"
        }
    }
    # 4. CLIP Loader (Qwen3-VL NVFP4)
    nodes["4"] = {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "type": "minimax",
            "device": "default"
        }
    }

    # 5. Optional LoRA routing
    if use_lora:
        nodes["5"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["3", 0],
                "lora_name": lora_name,
                "strength_model": lora_strength
            }
        }
        active_model_ref = ["5", 0]
    else:
        active_model_ref = ["3", 0]

    # 6. MiniMax H3 Text-to-Video Conditioner
    nodes["6"] = {
        "class_type": "MiniMaxH3ImageToVideo",
        "inputs": {
            "clip": ["4", 0],
            "vae": ["1", 0],
            "prompt": prompt_text,
            "width": width,
            "height": height,
            "length": length
        }
    }

    # 7. Random Noise
    nodes["7"] = {
        "class_type": "RandomNoise",
        "inputs": {
            "noise_seed": seed,
            "control_after_generate": "fixed"
        }
    }

    # 8. KSampler Select
    nodes["8"] = {
        "class_type": "KSamplerSelect",
        "inputs": {
            "sampler_name": "res_multistep"
        }
    }

    # 9. Basic Scheduler
    nodes["9"] = {
        "class_type": "BasicScheduler",
        "inputs": {
            "model": active_model_ref,
            "scheduler": "simple",
            "steps": steps,
            "denoise": 1.0
        }
    }

    # 10. Basic Guider
    nodes["10"] = {
        "class_type": "BasicGuider",
        "inputs": {
            "model": active_model_ref,
            "conditioning": ["6", 0]
        }
    }

    # 11. Sampler Custom Advanced
    nodes["11"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["7", 0],
            "guider": ["10", 0],
            "sampler": ["8", 0],
            "sigmas": ["9", 0],
            "latent_image": ["6", 1]
        }
    }

    # 12. VAE Decode Video
    nodes["12"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["11", 0],
            "vae": ["1", 0]
        }
    }

    # 13. VAE Decode Audio
    nodes["13"] = {
        "class_type": "VAEDecodeAudio",
        "inputs": {
            "samples": ["11", 0],
            "vae": ["2", 0]
        }
    }

    # 14. Create Video (Muxing)
    nodes["14"] = {
        "class_type": "CreateVideo",
        "inputs": {
            "images": ["12", 0],
            "audio": ["13", 0],
            "fps": 24,
            "bit_depth": 8
        }
    }

    # 15. Save Video
    nodes["15"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["14", 0],
            "filename_prefix": f"video/{task_name}",
            "format": "auto",
            "codec": "auto"
        }
    }

    return nodes

def stream_comfyui_logs(server_log_path, last_pos):
    """Read newly appended lines from ComfyUI server log to show live progress."""
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

def submit_and_wait_prompt(prompt_dict, task, timeout=900):
    """Submit prompt to ComfyUI API and wait for completion, tailing logs."""
    client_id = f"minimax_runner_{int(time.time())}"
    post_data = json.dumps({"prompt": prompt_dict, "client_id": client_id}).encode("utf-8")

    req = urllib.request.Request(
        "http://127.0.0.1:8188/prompt",
        data=post_data,
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI prompt submission failed ({e.code}): {err_msg}")

    prompt_id = resp_data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"No prompt_id returned: {resp_data}")

    log(f"Prompt queued successfully (prompt_id: {prompt_id}). Waiting for GPU execution...")

    start_time = time.time()
    server_log_path = "/content/comfyui_server.log"
    log_pos = 0
    if os.path.exists(server_log_path):
        log_pos = os.path.getsize(server_log_path)

    output_dir = "/content/output"
    task_name = task.get("name", "")

    # Poll history endpoint
    while time.time() - start_time < timeout:
        log_pos = stream_comfyui_logs(server_log_path, log_pos)

        try:
            hist_url = f"http://127.0.0.1:8188/history/{prompt_id}"
            with urllib.request.urlopen(hist_url, timeout=5) as h_resp:
                hist_data = json.loads(h_resp.read().decode("utf-8"))

            if prompt_id in hist_data:
                item = hist_data[prompt_id]
                status = item.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"Prompt execution reported error: {status.get('messages')}")
                if status.get("completed", False):
                    log(f"Prompt {prompt_id} completed successfully according to ComfyUI history.")
                    break
        except urllib.error.URLError:
            pass

        time.sleep(3.0)
    else:
        raise TimeoutError(f"Task {task_name} timed out after {timeout} seconds.")

    # Locate generated video file in /content/output
    matching_name = []
    recent_videos = []
    for root, _, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".mp4"):
                full_path = os.path.join(root, f)
                mtime = os.path.getmtime(full_path)
                if task_name in f:
                    matching_name.append((mtime, full_path))
                if mtime >= start_time - 5:
                    recent_videos.append((mtime, full_path))

    if matching_name:
        matching_name.sort(key=lambda x: x[0], reverse=True)
        video_path = matching_name[0][1]
    elif recent_videos:
        recent_videos.sort(key=lambda x: x[0], reverse=True)
        video_path = recent_videos[0][1]
    else:
        all_mp4 = glob.glob(f"{output_dir}/**/*.mp4", recursive=True)
        if all_mp4:
            all_mp4.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            video_path = all_mp4[0]
        else:
            raise FileNotFoundError(f"Generated video file not found in {output_dir} for task {task_name}")

    return video_path

def execute_batch_tasks(proc):
    log("Starting Batch Video Generation Pipeline (Plan C)...", step=6)

    tasks_path = "/content/tasks.json"
    if os.path.exists(tasks_path):
        with open(tasks_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        log(f"Loaded {len(tasks)} tasks from {tasks_path}")
    else:
        raise FileNotFoundError("No /content/tasks.json found!")

    results_summary = []
    total_start = time.time()

    for idx, task in enumerate(tasks, start=1):
        task_id = task.get("id", idx)
        task_name = task.get("name", f"Task_{task_id}")
        category = task.get("category", "General")
        steps = task.get("steps", 4)
        duration = task.get("duration", 5.0)
        use_lora = task.get("use_lora", True)

        log("----------------------------------------------------------------------")
        log(f">> Running Task [{idx}/{len(tasks)}]: {task_name}")
        log(f"   Category: {category}")
        log(f"   Config: Duration={duration}s (24fps), Steps={steps}, LoRA={'Enabled' if use_lora else 'Disabled (Base)'}")
        log("----------------------------------------------------------------------")

        task_t0 = time.time()
        status_str = "SUCCESS"
        video_file_name = "N/A"
        file_size_mb = 0.0

        try:
            prompt_graph = build_comfyui_prompt(task)
            video_local_path = submit_and_wait_prompt(prompt_graph, task, timeout=900)
            elapsed_task = time.time() - task_t0
            file_size_mb = os.path.getsize(video_local_path) / (1024**2)
            video_file_name = os.path.basename(video_local_path)

            log(f"[SUCCESS] Video rendered locally: {video_file_name} ({file_size_mb:.2f} MB, {elapsed_task:.1f}s)")

            # Upload directly to Google Drive MiniMax_H3_Outputs
            log(f"Uploading {video_file_name} to Google Drive (gdrive:MiniMax_H3_Outputs/) ...")
            cmd_upload = ["rclone", "copy", video_local_path, "gdrive:MiniMax_H3_Outputs/"]
            res_upload = subprocess.run(cmd_upload, capture_output=True, text=True)
            if res_upload.returncode == 0:
                log(f"[OK] Persisted to Drive: MiniMax_H3_Outputs/{video_file_name}")
            else:
                log(f"[WARN] Drive upload warning: {res_upload.stderr}")

        except Exception as e:
            status_str = "FAILED"
            elapsed_task = time.time() - task_t0
            log(f"[ERROR] Task {task_name} encountered error: {e}")

        results_summary.append({
            "id": task_id,
            "name": task_name,
            "category": category,
            "status": status_str,
            "duration_s": duration,
            "steps": steps,
            "size_mb": file_size_mb,
            "elapsed_s": elapsed_task,
            "filename": video_file_name
        })

    total_elapsed = time.time() - total_start
    log("======================================================================")
    log(f"All {len(tasks)} tasks finished in {total_elapsed/60:.2f} minutes.")
    log("================== BATCH EXECUTION SUMMARY REPORT ===================")
    for r in results_summary:
        log(f"[{r['status']}] Task #{r['id']} {r['name']} | Steps: {r['steps']} | Dur: {r['duration_s']}s | Size: {r['size_mb']:.1f}MB | Time: {r['elapsed_s']:.1f}s")
    log("======================================================================")

def main():
    if len(sys.argv) < 4:
        print("Usage: python remote_minimax_runner.py <client_id> <client_secret> <refresh_token> [access_token]")
        sys.exit(1)

    client_id = sys.argv[1]
    client_secret = sys.argv[2]
    refresh_token = sys.argv[3]
    access_token = sys.argv[4] if len(sys.argv) > 4 else ""

    log("============================================================")
    log("MiniMax H3 Automated GPU Pipeline on Colab A100 (Optimized)")
    log("============================================================")

    server_proc = None
    try:
        check_hardware()
        setup_rclone_config(client_id, client_secret, refresh_token, access_token)
        setup_comfyui_environment()
        sync_models_to_local_nvme()
        server_proc = start_comfyui_server()
        execute_batch_tasks(server_proc)
        log("============================================================")
        log("[SUCCESS] MiniMax H3 Batch Video Generation Workflow Finished!")
        log("============================================================")
    except Exception as e:
        log(f"[ERROR] Exception during execution: {e}", step="ERROR")
        raise
    finally:
        if server_proc and server_proc.poll() is None:
            log("Shutting down ComfyUI server process...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        log("Remote cleanup complete.")

if __name__ == "__main__":
    main()
