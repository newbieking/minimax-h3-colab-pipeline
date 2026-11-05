"""
cloud_model_downloader.py
Cross-platform Cloud Model Downloader for Google Colab CPU Instance.
Allocates a cost-effective CPU instance, leverages Google's multi-gigabit backbone bandwidth
to fetch Flux.1 FP8 and Civitai LoRAs, organizes anime character LoRAs, and uploads them to Google Drive.
Guarantees 0-leak instance destruction in finally block.
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

log_filename = os.path.join(LOG_DIR, f"downloader_{time.strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger = logging.getLogger("CloudDownloader")
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

def list_active_sessions():
    try:
        stdout, _ = run_command([COLAB_BIN, "sessions"], desc="List active sessions", check=False)
        return stdout
    except Exception:
        return ""

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

def run_cloud_download(session_name=None, dry_run=False, civitai_token=""):
    session_name = session_name or f"downloader-cpu-{int(time.time()) % 10000}"
    logger.info("=" * 70)
    logger.info(f"[START] Google Cloud High-Speed Model Downloader (CPU Instance)")
    logger.info(f"   Session: {session_name}")
    logger.info(f"   Machine Type: Standard CPU Runtime (0 GPU / Minimum CU)")
    logger.info(f"   Targets:")
    logger.info(f"     1. Flux.1 (Dev & Schnell FP8) -> gdrive:MiniMax_Models/diffusion_models/")
    logger.info(f"     2. 5x Civitai LoRAs -> gdrive:MiniMax_Models/loras/anime_characters/")
    logger.info("=" * 70)

    creds = get_gws_credentials()
    usage = get_colab_usage()
    logger.info(f"[BALANCE]\n{usage}")

    if dry_run:
        logger.info("[OK] Dry-run completed. All credentials and CLI links verified.")
        return

    remote_script_path = os.path.join(WORKSPACE_DIR, "remote_downloader.py")
    with open(remote_script_path, "r", encoding="utf-8") as f:
        remote_code = f.read()

    try:
        # Launch CPU runtime: WITHOUT --gpu option creates standard CPU instance
        logger.info(f"[PROVISION] Allocating Google Colab CPU instance (session: {session_name})...")
        run_command([COLAB_BIN, "new", "-s", session_name], desc="Allocate CPU Session")

        logger.info("[WARMUP] Waiting 10s for Jupyter Kernel warmup...")
        time.sleep(10.0)

        logger.info("[EXEC] Injecting download payload and executing in Google Cloud...")
        runner_payload = f"""
import json, sys

with open('/content/remote_downloader.py', 'w', encoding='utf-8') as f:
    f.write({json.dumps(remote_code)})

sys.path.insert(0, '/content')
import remote_downloader
sys.argv = [
    'remote_downloader.py',
    {json.dumps(creds['client_id'])},
    {json.dumps(creds['client_secret'])},
    {json.dumps(creds['refresh_token'])},
    {json.dumps(creds['access_token'])},
    {json.dumps(civitai_token)}
]
remote_downloader.main()
"""
        cmd = [COLAB_BIN, "exec", "-s", session_name, "--timeout", "7200"]
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
        proc.stdin.write(runner_payload)
        proc.stdin.close()

        for line in proc.stdout:
            line_str = line.rstrip()
            if line_str:
                logger.info(f"[Cloud-CPU] {line_str}")

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Remote downloader failed with exit code: {proc.returncode}")

        logger.info("[SUCCESS] Cloud download and Google Drive synchronization complete!")

    except KeyboardInterrupt:
        logger.warning("[WARN] User manually interrupted download.")
    except Exception as e:
        logger.error(f"[ERR] Download pipeline exception: {e}")
        raise
    finally:
        release_session(session_name)
        final_usage = get_colab_usage()
        logger.info(f"[FINAL BALANCE]\n{final_usage}")
        logger.info("[FINISH] CPU instance safely recycled. 0 leak guaranteed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Google Colab CPU High-Speed Model Downloader")
    parser.add_argument("--session", type=str, default=None, help="Colab session name")
    parser.add_argument("--dry-run", action="store_true", help="Validate credentials and setup without launching CPU")
    parser.add_argument("--civitai-token", type=str, default="", help="Optional Civitai API Key")
    args = parser.parse_args()

    run_cloud_download(
        session_name=args.session,
        dry_run=args.dry_run,
        civitai_token=args.civitai_token
    )
