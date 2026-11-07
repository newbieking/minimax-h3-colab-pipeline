"""
universal_batch_scheduler.py
Content-Agnostic, High-Efficiency Batch Pipeline Scheduler for Colab GPU.
Key Capabilities:
- Single Cold-Start, Multi-Task Execution: Executes entire task manifest in one session to minimize startup overhead.
- Universal Task Interface: Works with any tasks JSON (Lookbook, Series, Concepts).
- Foolproof 0-Leak Protection: Enforces strict session termination in finally blocks.
- Automatic Local Ingestion: Uses relative-path gws downloads to pull outputs for agent visual QC.
"""

import os
import sys
import time
import json
import shutil
import logging
import argparse
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

log_filename = os.path.join(LOG_DIR, f"universal_batch_{time.strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger = logging.getLogger("UniversalBatch")
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
    logger.info(f"[STOP] Terminating Colab GPU session: {session_name} ...")
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

    req = urllib.request.Request(token_url, data=post_data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req) as resp:
        token_info = json.loads(resp.read().decode("utf-8"))
        creds["access_token"] = token_info.get("access_token", "")

    return creds

def run_batch_pipeline(tasks_path, dest_drive_folder, local_preview_dir, gpu_type="a100", timeout_sec=1200):
    logger.info("================================================================")
    logger.info("  STARTING UNIVERSAL BATCH COLAB PIPELINE (SINGLE COLD-START)")
    logger.info(f"  Tasks: {tasks_path}")
    logger.info(f"  Drive Target: {dest_drive_folder}")
    logger.info("================================================================")

    init_usage = get_colab_usage()
    logger.info(f"[QUOTA] Initial Colab Status:\n{init_usage}")

    session_name = f"univ_batch_{int(time.time())}"
    creds = get_gws_credentials()

    with open(os.path.join(WORKSPACE_DIR, "remote_flux_universal_runner.py"), "r", encoding="utf-8") as f:
        runner_code = f.read()

    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks_json_str = f.read()

    tasks_count = len(json.loads(tasks_json_str))
    logger.info(f"[PLAN] Queued {tasks_count} tasks in a single persistent GPU execution session.")

    session_started = False
    try:
        logger.info(f"[PROVISION] Allocating Colab {gpu_type} GPU runtime (session: {session_name})...")
        run_command([COLAB_BIN, "new", "-s", session_name, "--gpu", gpu_type], desc=f"Allocate {gpu_type} GPU Session")
        session_started = True

        logger.info("[WARMUP] Waiting 10s for Jupyter Kernel warmup...")
        time.sleep(10.0)

        logger.info("[EXEC] Injecting Universal Runner & Task Manifest into Colab...")
        runner_payload = f"""
import json, sys

with open('/content/remote_flux_universal_runner.py', 'w', encoding='utf-8') as f:
    f.write({json.dumps(runner_code)})

with open('/content/tasks.json', 'w', encoding='utf-8') as f:
    f.write({json.dumps(tasks_json_str)})

sys.path.insert(0, '/content')
import remote_flux_universal_runner
sys.argv = [
    'remote_flux_universal_runner.py',
    {json.dumps(creds['client_id'])},
    {json.dumps(creds['client_secret'])},
    {json.dumps(creds['refresh_token'])},
    {json.dumps(creds['access_token'])},
    '/content/tasks.json',
    {json.dumps(dest_drive_folder)}
]
remote_flux_universal_runner.main()
"""
        colab_env = os.environ.copy()
        colab_env["PYTHONIOENCODING"] = "utf-8"
        colab_env["PYTHONUTF8"] = "1"

        cmd = [COLAB_BIN, "exec", "-s", session_name, "--timeout", str(timeout_sec)]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=colab_env
        )
        proc.stdin.write(runner_payload)
        proc.stdin.close()

        for line in proc.stdout:
            clean_line = line.strip()
            if clean_line:
                logger.info(f"[REMOTE] {clean_line}")

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Remote execution failed with exit code: {proc.returncode}")

        logger.info("[SUCCESS] Remote Universal Batch generation completed!")

    finally:
        if session_started:
            release_session(session_name)
            time.sleep(3.0)
            final_usage = get_colab_usage()
            logger.info(f"[QUOTA] Final Colab Status:\n{final_usage}")

def download_results_locally(dest_drive_folder, local_preview_dir):
    logger.info(f"[SYNC] Fetching outputs from {dest_drive_folder} to {local_preview_dir} ...")
    os.chdir(WORKSPACE_DIR)
    os.makedirs(local_preview_dir, exist_ok=True)

    folder_name = dest_drive_folder.rstrip("/").split("/")[-1]
    search_params = json.dumps({
        "q": f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
        "fields": "files(id, name)"
    })
    res = subprocess.check_output([GWS_BIN, "drive", "files", "list", "--params", search_params], text=True, encoding="utf-8")
    data = json.loads(res)
    folders = data.get("files", [])
    if not folders:
        logger.warning(f"[WARN] Folder {folder_name} not found yet in Drive.")
        return

    folder_id = folders[0]["id"]
    file_params = json.dumps({
        "q": f"'{folder_id}' in parents and trashed = false",
        "fields": "files(id, name, mimeType)"
    })
    res_files = subprocess.check_output([GWS_BIN, "drive", "files", "list", "--params", file_params], text=True, encoding="utf-8")
    files_data = json.loads(res_files)

    for item in files_data.get("files", []):
        file_id = item["id"]
        file_name = item["name"]
        dest_rel = f"{local_preview_dir}/{file_name}"
        logger.info(f"[DOWNLOAD] Downloading {file_name} -> {dest_rel} ...")
        params = json.dumps({"fileId": file_id, "alt": "media"})
        cmd = [GWS_BIN, "drive", "files", "get", "--params", params, "-o", dest_rel]
        subprocess.run(cmd, check=False)

    logger.info(f"[SYNC COMPLETE] All assets downloaded to {local_preview_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Colab Batch Scheduler")
    parser.add_argument("--tasks", default="flux_combined_tasks.json", help="Path to tasks JSON")
    parser.add_argument("--dest-drive", default="gdrive:Cloud_AI_Workspace/Outputs/Images/FLUX_Anime_Lookbook/", help="Google Drive destination")
    parser.add_argument("--preview-dir", default="flux_anime_lookbook_preview", help="Local preview directory relative to workspace")
    parser.add_argument("--gpu", default="a100", help="GPU accelerator type")
    parser.add_argument("--timeout", type=int, default=1200, help="Session timeout in seconds")
    args = parser.parse_args()

    tasks_abs = os.path.abspath(args.tasks) if os.path.isabs(args.tasks) else os.path.join(WORKSPACE_DIR, args.tasks)

    try:
        run_batch_pipeline(tasks_abs, args.dest_drive, args.preview_dir, gpu_type=args.gpu, timeout_sec=args.timeout)
    except Exception as e:
        logger.error(f"[FATAL] Pipeline failed: {e}", exc_info=True)
    finally:
        download_results_locally(args.dest_drive, args.preview_dir)
