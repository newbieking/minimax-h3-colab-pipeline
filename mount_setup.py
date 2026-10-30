"""
mount_setup.py
Configures and launches rclone mount on Google Colab instance using gws credentials.
"""

import os
import sys
import time
import json
import subprocess

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def setup_rclone_and_mount(client_id, client_secret, refresh_token):
    log("Installing rclone and fuse3...")
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "rclone", "fuse3"], check=True)

    rclone_config_dir = os.path.expanduser("~/.config/rclone")
    os.makedirs(rclone_config_dir, exist_ok=True)
    config_path = os.path.join(rclone_config_dir, "rclone.conf")

    token_json = json.dumps({
        "access_token": "",
        "token_type": "Bearer",
        "refresh_token": refresh_token,
        "expiry": "2020-01-01T00:00:00Z"
    })

    config_content = f"""[gdrive]
type = drive
client_id = {client_id}
client_secret = {client_secret}
scope = drive
token = {token_json}
"""
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)
    log(f"Wrote rclone config to {config_path}")

    mount_dir = "/content/drive"
    os.makedirs(mount_dir, exist_ok=True)

    log(f"Mounting gdrive: at {mount_dir} via rclone...")
    mount_cmd = [
        "rclone", "mount", "gdrive:", mount_dir,
        "--vfs-cache-mode", "minimal",
        "--dir-cache-time", "1000h",
        "--poll-interval", "1m",
        "--read-only",
        "--allow-other",
        "--daemon"
    ]
    res = subprocess.run(mount_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log(f"Daemon mount returned {res.returncode}, stderr: {res.stderr}")

    log("Waiting for mount point to become available...")
    for i in range(15):
        if os.path.ismount(mount_dir) or os.path.exists(os.path.join(mount_dir, "MiniMax_Models")) or os.path.exists(os.path.join(mount_dir, "MyDrive")):
            log(f"[OK] Mount is LIVE in {i+1}s!")
            break
        time.sleep(1.0)
    else:
        log("Mount check warning, checking root entries...")

    try:
        entries = os.listdir(mount_dir)
        log(f"Visible entries in {mount_dir}: {entries}")
        # Locate MiniMax_Models
        if "MiniMax_Models" in entries:
            models_path = os.path.join(mount_dir, "MiniMax_Models")
        elif os.path.exists(os.path.join(mount_dir, "MyDrive", "MiniMax_Models")):
            models_path = os.path.join(mount_dir, "MyDrive", "MiniMax_Models")
        else:
            models_path = None

        if models_path:
            log(f"[SUCCESS] MiniMax_Models located at: {models_path}")
            log(f"Subdirectories: {os.listdir(models_path)}")
        else:
            log("[WARN] MiniMax_Models not immediately visible in root.")
    except Exception as e:
        log(f"Error inspecting mount dir: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python mount_setup.py <client_id> <client_secret> <refresh_token>")
        sys.exit(1)
    setup_rclone_and_mount(sys.argv[1], sys.argv[2], sys.argv[3])
