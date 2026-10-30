"""
test_runner_cpu.py
Tests remote_minimax_runner mount logic on free CPU session.
"""

import os
import sys
import json
import subprocess

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def get_gws_credentials():
    out = subprocess.check_output(["gws.cmd", "auth", "export"], text=True, encoding="utf-8", errors="replace")
    for line in out.splitlines():
        if line.strip().startswith("{") and line.strip().endswith("}"):
            return json.loads(line.strip())
    return json.loads(out.strip())

def main():
    creds = get_gws_credentials()
    client_id = creds["client_id"]
    client_secret = creds["client_secret"]
    refresh_token = creds["refresh_token"]

    session_name = "test-fuse-cpu"
    print(f"Allocating free CPU session: {session_name}...")
    try:
        subprocess.run(["colab.exe", "new", "-s", session_name], check=True)

        runner_payload = f"""
import subprocess, os, json, time

subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "rclone", "fuse3"], check=True)
subprocess.run("echo user_allow_other >> /etc/fuse.conf", shell=True)

rclone_config_dir = os.path.expanduser("~/.config/rclone")
os.makedirs(rclone_config_dir, exist_ok=True)
config_path = os.path.join(rclone_config_dir, "rclone.conf")

token_json = json.dumps({{
    "access_token": "",
    "token_type": "Bearer",
    "refresh_token": {json.dumps(refresh_token)},
    "expiry": "2020-01-01T00:00:00Z"
}})

config_content = f\"\"\"[gdrive]
type = drive
client_id = {client_id}
client_secret = {client_secret}
scope = drive
token = {{token_json}}
\"\"\"
with open(config_path, "w", encoding="utf-8") as f:
    f.write(config_content)

mount_dir = "/content/drive"
os.makedirs(mount_dir, exist_ok=True)

# Unmount if already mounted
subprocess.run(["fusermount", "-u", mount_dir], stderr=subprocess.DEVNULL)

cmd = [
    "rclone", "mount", "gdrive:", mount_dir,
    "--vfs-cache-mode", "minimal",
    "--dir-cache-time", "1000h",
    "--read-only",
    "--daemon"
]
res = subprocess.run(cmd, capture_output=True, text=True)
print("rclone returncode:", res.returncode)
print("rclone stdout:", res.stdout)
print("rclone stderr:", res.stderr)

for i in range(10):
    if os.path.exists(os.path.join(mount_dir, "MiniMax_Models")) or os.path.exists(os.path.join(mount_dir, "MyDrive")):
        print(f"SUCCESS: Mount active in {{i+1}}s!")
        break
    time.sleep(1.0)

print("Listing /content/drive:", os.listdir(mount_dir) if os.path.exists(mount_dir) else "not exist")
"""
        cmd = ["colab.exe", "exec", "-s", session_name, "--timeout", "300"]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        stdout, _ = proc.communicate(input=runner_payload)

        print("--- Output from VM ---")
        print(stdout)
        print("----------------------")
    finally:
        print(f"Releasing {session_name}...")
        subprocess.run(["colab.exe", "stop", "-s", session_name], check=False)
        print("Done.")

if __name__ == "__main__":
    main()
