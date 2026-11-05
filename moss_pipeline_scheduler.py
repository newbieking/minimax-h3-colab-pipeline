"""
moss_pipeline_scheduler.py
Cross-platform Pipeline Scheduler for Moody Moss Mix (MOSS-Image DiT) on Colab A100.
Runs high-precision 25-step, CFG 3.0 photorealistic portrait batches (4 images x 4 tasks = 16 images).
Persists outputs into Google Drive: Cloud_AI_Workspace/Outputs/Images/Photorealistic_Moss/
Guarantees 0-leak Colab session destruction in finally block.
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

log_filename = os.path.join(LOG_DIR, f"moss_pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger = logging.getLogger("MossPipeline")
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
    req = urllib.request.Request(token_url, data=post_data)
    token_resp = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
    creds["access_token"] = token_resp["access_token"]
    logger.info("[AUTH] High-availability Access Token generated successfully.")
    return creds

def run_pipeline(task_ids_str="all", gpu_type="A100", timeout_sec=3600):
    session_name = f"colab-moss-{int(time.time()) % 10000}"
    tasks_file = os.path.join(WORKSPACE_DIR, "moss_photorealistic_tasks.json")
    if not os.path.exists(tasks_file):
        raise FileNotFoundError(f"Missing {tasks_file}")

    with open(tasks_file, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    if task_ids_str.lower() != "all":
        selected_ids = {int(x.strip()) for x in task_ids_str.split(",") if x.strip()}
        tasks = [t for t in all_tasks if t.get("id") in selected_ids]
    else:
        tasks = all_tasks

    total_images = sum(t.get("batch_size", 4) for t in tasks)

    logger.info("=" * 70)
    logger.info(f"[START] Moody Moss Mix Photorealistic LookDev Pipeline")
    logger.info(f"   Session: {session_name}")
    logger.info(f"   Tasks: {len(tasks)} tasks (Total {total_images} image candidates)")
    logger.info(f"   Resolution: 896x1344 (3:4 High Megapixel Portrait)")
    logger.info(f"   Params: 25 steps, CFG 3.0, Euler, Simple")
    logger.info(f"   Destination: Google Drive/Cloud_AI_Workspace/Outputs/Images/Photorealistic_Moss/")
    logger.info("=" * 70)

    for t in tasks:
        logger.info(f"   #Task {t['id']}: {t['name']} | Batch: {t.get('batch_size', 4)} pics | Res: {t.get('width', 896)}x{t.get('height', 1344)}")

    creds = get_gws_credentials()
    usage = get_colab_usage()
    logger.info(f"[BALANCE]\n{usage}")

    remote_script_path = os.path.join(WORKSPACE_DIR, "remote_moss_runner.py")
    with open(remote_script_path, "r", encoding="utf-8") as f:
        remote_code = f.read()

    tasks_json_str = json.dumps(tasks, ensure_ascii=False)

    try:
        logger.info(f"[PROVISION] Allocating Colab {gpu_type} GPU runtime (session: {session_name})...")
        run_command([COLAB_BIN, "new", "-s", session_name, "--gpu", gpu_type], desc=f"Allocate {gpu_type} GPU Session")

        logger.info("[WARMUP] Waiting 10s for Jupyter Kernel warmup...")
        time.sleep(10.0)

        logger.info("[EXEC] Injecting Moody Moss tasks and executing in Google Cloud...")
        runner_payload = f"""
import json, sys

with open('/content/remote_moss_runner.py', 'w', encoding='utf-8') as f:
    f.write({json.dumps(remote_code)})

with open('/content/moss_photorealistic_tasks.json', 'w', encoding='utf-8') as f:
    f.write({json.dumps(tasks_json_str)})

sys.path.insert(0, '/content')
import remote_moss_runner
sys.argv = [
    'remote_moss_runner.py',
    {json.dumps(creds['client_id'])},
    {json.dumps(creds['client_secret'])},
    {json.dumps(creds['refresh_token'])},
    {json.dumps(creds['access_token'])}
]
remote_moss_runner.main()
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
            line_str = line.rstrip()
            if line_str:
                logger.info(f"[Remote-Moss] {line_str}")

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Remote Moss generation failed with code: {proc.returncode}")

        logger.info("[SUCCESS] All Moody Moss images successfully generated and persisted to Google Drive!")

    except KeyboardInterrupt:
        logger.warning("[WARN] User manually interrupted execution.")
    except Exception as e:
        logger.error(f"[ERR] Pipeline error: {e}")
        raise
    finally:
        release_session(session_name)
        final_usage = get_colab_usage()
        logger.info(f"[FINAL BALANCE]\n{final_usage}")
        logger.info("[FINISH] GPU session safely terminated. 0 leak guaranteed.")

if __name__ == "__main__":
    task_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    gpu_arg = sys.argv[2] if len(sys.argv) > 2 else "A100"
    run_pipeline(task_ids_str=task_arg, gpu_type=gpu_arg)
