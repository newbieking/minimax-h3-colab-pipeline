"""
remote_downloader.py
High-speed cloud model downloader executed inside Google Colab (CPU instance).
Leverages Google datacenter multi-gigabit backbone bandwidth with aria2 multi-connection download,
and automatically persists models into Google Drive via rclone.
"""

import os
import sys
import time
import json
import subprocess
import shutil

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def setup_environment(client_id, client_secret, refresh_token, access_token=""):
    log("Setting up rclone, aria2 and dependencies...")
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "rclone", "aria2"], check=True)

    # Configure rclone
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

def download_file_aria2(url, dest_dir, filename=None, headers=None):
    os.makedirs(dest_dir, exist_ok=True)
    cmd = [
        "aria2c",
        "--console-log-level=warn",
        "--summary-interval=5",
        "-x", "16",
        "-s", "16",
        "-k", "1M",
        "--dir", dest_dir,
        url
    ]
    if filename:
        cmd.extend(["-o", filename])
    if headers:
        for h in headers:
            cmd.extend(["--header", h])

    log(f"Downloading: {filename or url} ...")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        log(f"[WARN] aria2c failed with code {res.returncode}, falling back to curl...")
        curl_cmd = ["curl", "-L", "-C", "-", url]
        if filename:
            curl_cmd.extend(["-o", os.path.join(dest_dir, filename)])
        if headers:
            for h in headers:
                curl_cmd.extend(["-H", h])
        subprocess.run(curl_cmd, check=True)

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 remote_downloader.py <client_id> <client_secret> <refresh_token> [access_token] [civitai_token]")
        sys.exit(1)

    client_id = sys.argv[1]
    client_secret = sys.argv[2]
    refresh_token = sys.argv[3]
    access_token = sys.argv[4] if len(sys.argv) > 4 else ""
    civitai_token = sys.argv[5] if len(sys.argv) > 5 else ""

    setup_environment(client_id, client_secret, refresh_token, access_token)

    work_dir = "/content/downloads"
    os.makedirs(work_dir, exist_ok=True)

    civitai_headers = []
    if civitai_token:
        civitai_headers.append(f"Authorization: Bearer {civitai_token}")
    civitai_headers.append("User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

    # 1. Civitai LoRA tasks (Target: MiniMax_Models/loras/anime_characters/)
    anime_loras_dir = os.path.join(work_dir, "anime_characters")
    os.makedirs(anime_loras_dir, exist_ok=True)

    lora_tasks = [
        {
            "name": "genshinfull1-000006.safetensors",
            "desc": "原神全角色 All-In-One LoRA (v1.0)",
            "url": "https://civitai.com/api/download/models/116970"
        },
        {
            "name": "jirai_v2.safetensors",
            "desc": "地雷系/量产型服饰风格 LoRA (v2.0)",
            "url": "https://civitai.com/api/download/models/33010"
        },
        {
            "name": "BArtstyle-KREA2-V2-1024.safetensors",
            "desc": "蔚蓝档案画风模型 (Blue Archive Art Style Krea2-V2)",
            "url": "https://civitai.com/api/download/models/3220090"
        },
        {
            "name": "add_detail.safetensors",
            "desc": "细节调整 LoRA (Detail Tweaker v1.0)",
            "url": "https://civitai.com/api/download/models/62833"
        },
        {
            "name": "ClothingAdjuster3.safetensors",
            "desc": "LEOSAM 衣物增减 LoRA (ClothingAdjuster 3.0)",
            "url": "https://civitai.com/api/download/models/117151"
        }
    ]

    log("============================================================")
    log("Step 1: Downloading 5 Anime Character LoRAs from Civitai...")
    log("============================================================")

    for lora in lora_tasks:
        log(f">> [LoRA] {lora['desc']} -> {lora['name']}")
        download_file_aria2(
            url=lora["url"],
            dest_dir=anime_loras_dir,
            filename=lora["name"],
            headers=civitai_headers
        )

    log("Uploading anime character LoRAs to Google Drive (gdrive:MiniMax_Models/loras/anime_characters/) ...")
    subprocess.run([
        "rclone", "copy",
        anime_loras_dir,
        "gdrive:MiniMax_Models/loras/anime_characters/",
        "--progress"
    ], check=True)
    log("[OK] All anime character LoRAs successfully synced to Google Drive!")

    # 2. Flux.1 Models (Target: MiniMax_Models/diffusion_models/)
    flux_dir = os.path.join(work_dir, "flux")
    os.makedirs(flux_dir, exist_ok=True)

    flux_models = [
        {
            "name": "flux1-dev-fp8.safetensors",
            "desc": "FLUX.1 [dev] FP8 (Direct Diffusion Model ~11.9 GB)",
            "url": "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors"
        },
        {
            "name": "flux1-schnell-fp8.safetensors",
            "desc": "FLUX.1 [schnell] FP8 (Fast 4-Step Model ~11.9 GB)",
            "url": "https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors"
        }
    ]

    log("============================================================")
    log("Step 2: Downloading Flux.1 (Dev & Schnell FP8) via Google Backbone...")
    log("============================================================")

    for model in flux_models:
        log(f">> [Flux.1] {model['desc']} -> {model['name']}")
        download_file_aria2(
            url=model["url"],
            dest_dir=flux_dir,
            filename=model["name"]
        )
        # Upload right after each large file to maximize disk & bandwidth pipeline
        log(f"Uploading {model['name']} to Google Drive (gdrive:MiniMax_Models/diffusion_models/) ...")
        subprocess.run([
            "rclone", "copy",
            os.path.join(flux_dir, model["name"]),
            "gdrive:MiniMax_Models/diffusion_models/",
            "--progress"
        ], check=True)
        # Free up local disk space on Colab
        os.remove(os.path.join(flux_dir, model["name"]))
        log(f"[OK] {model['name']} uploaded and local cache freed.")

    log("============================================================")
    log("[FINISH] All requested models downloaded and persisted to Google Drive!")
    log("============================================================")

if __name__ == "__main__":
    main()
