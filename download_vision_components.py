"""
download_vision_components.py
Downloads sigclip_vision_patch14_384.safetensors (816 MB) to Google Drive:
Cloud_AI_Workspace/Models/clip_vision/
Runs on Colab CPU (0 CU cost) with foolproof cleanup.
"""

import os
import sys
import time
import json
import shutil
import urllib.request
import urllib.parse
import subprocess

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))

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

def get_gws_credentials():
    out = subprocess.check_output([GWS_BIN, "auth", "export", "--unmasked"], text=True, encoding="utf-8")
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

def run_download():
    print("Exporting GWS credentials...")
    creds = get_gws_credentials()
    session_name = f"dl_vision_{int(time.time())}"

    print(f"Allocating Colab CPU session: {session_name} ...")
    subprocess.run([COLAB_BIN, "new", "-s", session_name], check=True)
    time.sleep(8.0)

    try:
        remote_script = f"""
import os, subprocess, json

# 1. Setup rclone
token_dict = {{
    "access_token": {json.dumps(creds.get('access_token', ''))},
    "token_type": "Bearer",
    "refresh_token": {json.dumps(creds['refresh_token'])},
    "expiry": "2026-11-07T08:00:00Z"
}}
rclone_conf = f\"\"\"[gdrive]
type = drive
client_id = {creds['client_id']}
client_secret = {creds['client_secret']}
scope = drive
token = {{json.dumps(token_dict)}}
\"\"\"
os.makedirs(os.path.expanduser("~/.config/rclone"), exist_ok=True)
with open(os.path.expanduser("~/.config/rclone/rclone.conf"), "w") as f:
    f.write(rclone_conf)

subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "rclone", "aria2"], check=True)

# 2. Download sigclip_vision_patch14_384.safetensors
url = "https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors"
local_dest = "/content/sigclip_vision_patch14_384.safetensors"
print("Downloading sigclip_vision_patch14_384 via aria2c...")
subprocess.run(["aria2c", "-x", "16", "-s", "16", "-k", "1M", "-o", "sigclip_vision_patch14_384.safetensors", url], check=True)

# 3. Push to Google Drive
drive_target = "gdrive:Cloud_AI_Workspace/Models/clip_vision/"
print(f"Pushing to Google Drive: {{drive_target}} ...")
subprocess.run(["rclone", "copy", local_dest, drive_target, "--progress"], check=True)
print("SUCCESS: Vision component persisted to Google Drive!")
"""
        cmd = [COLAB_BIN, "exec", "-s", session_name, "--timeout", "600"]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        proc.stdin.write(remote_script)
        proc.stdin.close()

        for line in proc.stdout:
            print(f"[REMOTE] {line.strip()}")
        proc.wait()
    finally:
        print(f"Terminating CPU session: {session_name} ...")
        subprocess.run([COLAB_BIN, "stop", "-s", session_name], check=False)
        print("Session stopped. 0 CU consumed.")

if __name__ == "__main__":
    run_download()
