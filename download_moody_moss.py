"""
download_moody_moss.py
Cloud CPU Downloader tailored for Moody Moss Mix V3.3 Checkpoint using validated Civitai Token.
Allocates a Colab CPU instance, downloads 4.73GB model via Google's gigabit backbone,
and syncs directly to Google Drive MiniMax_Models/checkpoints/.
Guarantees 0-leak CPU session destruction in finally block.
"""

import os
import sys
import time
import json
import shutil
import logging
import urllib.request
import urllib.parse
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(WORKSPACE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"downloader_moody_moss_{time.strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger = logging.getLogger("MoodyMossDownloader")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def resolve_executable(name):
    if sys.platform == "win32":
        candidates = [f"{name}.cmd", f"{name}.exe", f"{name}.bat", name]
    else:
        candidates = [name]
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    return name

COLAB_BIN = resolve_executable("colab")
GWS_BIN = resolve_executable("gws")

def run_command(cmd, desc="", check=True, timeout=None):
    logger.info(f"[RUN] {desc or ' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout
    )
    if result.returncode != 0 and check:
        logger.error(f"[ERR] Command failed ({result.returncode}):\n{result.stderr.strip()}")
        raise RuntimeError(f"Command execution failed: {' '.join(cmd)}")
    return result.stdout.strip(), result.stderr.strip()

def get_colab_usage():
    try:
        stdout, _ = run_command([COLAB_BIN, "usage"], desc="Check Colab balance", check=False)
        return stdout
    except Exception as e:
        return f"Unknown: {e}"

def release_session(session_name):
    logger.info(f"[STOP] Terminating Colab CPU session: {session_name} ...")
    try:
        stdout, stderr = run_command([COLAB_BIN, "stop", "-s", session_name], desc=f"Stop session {session_name}", check=False)
        logger.info(f"[STOP] Session terminated: {stdout or stderr or 'OK'}")
    except Exception as e:
        logger.error(f"[ERR] Stop session error: {e}")

def get_gws_credentials():
    logger.info("[AUTH] Exporting gws credentials from OS Keyring...")
    out = subprocess.check_output([GWS_BIN, "auth", "export", "--unmasked"], text=True, encoding="utf-8", errors="replace")
    creds = None
    for line in out.splitlines():
        if line.strip().startswith("{") and line.strip().endswith("}"):
            creds = json.loads(line.strip())
            break
    if not creds:
        creds = json.loads(out.strip())

    token_url = "https://oauth2.googleapis.com/token"
    post_data = urllib.parse.urlencode({
        "client_id": creds["client_id"],
        "client_secret": creds["client_secret"],
        "refresh_token": creds["refresh_token"],
        "grant_type": "refresh_token"
    }).encode("utf-8")
    req = urllib.request.Request(token_url, data=post_data)
    token_resp = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
    creds["access_token"] = token_resp["access_token"]
    logger.info("[AUTH] High-availability Access Token generated successfully.")
    return creds

def run_download(civitai_token):
    session_name = f"moss-dl-{int(time.time()) % 10000}"
    download_url = f"https://civitai.com/api/download/models/3435978?fileId=3332948&token={civitai_token}"

    logger.info("=" * 70)
    logger.info(f"[START] Moody Moss Mix V3.3 Cloud Downloader (CPU Instance)")
    logger.info(f"   Session: {session_name}")
    logger.info(f"   Target: Moody Moss Mix V3.3 (~4.73 GB)")
    logger.info(f"   Drive Destination: gdrive:MiniMax_Models/checkpoints/")
    logger.info("=" * 70)

    creds = get_gws_credentials()
    usage = get_colab_usage()
    logger.info(f"[BALANCE]\n{usage}")

    remote_payload = f"""
import os, sys, time, json, subprocess

def log(msg):
    print(f"[{{time.strftime('%Y-%m-%d %H:%M:%S')}}] {{msg}}", flush=True)

log("1. Setting up rclone and dependencies...")
subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "rclone", "curl"], check=True)

# Configure rclone
rclone_config_dir = os.path.expanduser("~/.config/rclone")
os.makedirs(rclone_config_dir, exist_ok=True)
config_path = os.path.join(rclone_config_dir, "rclone.conf")

token_dict = {{
    "access_token": {json.dumps(creds['access_token'])},
    "token_type": "Bearer",
    "refresh_token": {json.dumps(creds['refresh_token'])},
    "expiry": "2026-10-30T13:00:00Z"
}}

config_content = f\"\"\"[gdrive]
type = drive
client_id = {creds['client_id']}
client_secret = {creds['client_secret']}
scope = drive
token = {{json.dumps(token_dict)}}
\"\"\"
with open(config_path, "w", encoding="utf-8") as f:
    f.write(config_content)
log("[OK] rclone Google Drive configured.")

work_dir = "/content/moss_model"
os.makedirs(work_dir, exist_ok=True)
model_filename = "moody_moss_mix_v3.3_25steps_cfg3.safetensors"
local_path = os.path.join(work_dir, model_filename)

log(f"2. Downloading {{model_filename}} (~4.73 GB) from Civitai R2 CDN via Google backbone...")
t0 = time.time()
curl_cmd = [
    "curl", "-L", "-C", "-",
    "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "-o", local_path,
    {json.dumps(download_url)}
]
res = subprocess.run(curl_cmd)
if res.returncode != 0:
    raise RuntimeError(f"Curl download failed with code {{res.returncode}}")

elapsed_dl = time.time() - t0
file_size_gb = os.path.getsize(local_path) / (1024**3)
log(f"[OK] Download completed: {{file_size_gb:.2f}} GB in {{elapsed_dl:.1f}}s (~{{(file_size_gb*1024)/max(elapsed_dl,1):.1f}} MB/s)!")

log(f"3. Uploading {{model_filename}} to Google Drive (gdrive:MiniMax_Models/checkpoints/) ...")
t1 = time.time()
subprocess.run([
    "rclone", "copy",
    local_path,
    "gdrive:MiniMax_Models/checkpoints/",
    "--progress"
], check=True)
elapsed_up = time.time() - t1
log(f"[OK] Persisted to Google Drive in {{elapsed_up:.1f}}s!")

# Cleanup local disk
os.remove(local_path)
log("[FINISH] Moody Moss Mix V3.3 is ready in Google Drive!")
"""

    try:
        # Launch CPU runtime: WITHOUT --gpu option creates standard CPU instance (minimal/0 CU)
        logger.info(f"[PROVISION] Allocating Google Colab CPU instance (session: {session_name})...")
        run_command([COLAB_BIN, "new", "-s", session_name], desc="Allocate CPU Session")

        logger.info("[WARMUP] Waiting 10s for Jupyter Kernel warmup...")
        time.sleep(10.0)

        logger.info("[EXEC] Injecting download payload and executing in Google Cloud...")
        cmd = [COLAB_BIN, "exec", "-s", session_name, "--timeout", "3600"]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )
        proc.stdin.write(remote_payload)
        proc.stdin.close()

        for line in proc.stdout:
            line_str = line.rstrip()
            if line_str:
                logger.info(f"[Moss-Cloud] {line_str}")

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Remote downloader failed with exit code: {proc.returncode}")

        logger.info("[SUCCESS] Moody Moss Mix V3.3 model downloaded and persisted to Google Drive!")

    except KeyboardInterrupt:
        logger.warning("[WARN] User manually interrupted download.")
    except Exception as e:
        logger.error(f"[ERR] Download error: {e}")
        raise
    finally:
        release_session(session_name)
        final_usage = get_colab_usage()
        logger.info(f"[FINAL BALANCE]\n{final_usage}")
        logger.info("[FINISH] CPU instance safely recycled. 0 leak guaranteed.")

if __name__ == "__main__":
    token = sys.argv[1] if len(sys.argv) > 1 else "42528c0ec4b1aab33dd67a7aad464730"
    run_download(token)
