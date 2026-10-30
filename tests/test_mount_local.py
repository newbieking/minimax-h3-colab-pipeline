"""
test_mount_local.py
Validates rclone transparent mount on a free Colab CPU session before touching GPU.
"""

import os
import sys
import json
import time
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
    print("1. Extracting local gws credentials...")
    creds = get_gws_credentials()
    client_id = creds["client_id"]
    client_secret = creds["client_secret"]
    refresh_token = creds["refresh_token"]
    print("   [OK] Credentials extracted from local keyring.")

    session_name = "test-rclone-cpu"
    print(f"2. Allocating free CPU session: {session_name} (0 CU cost)...")
    try:
        subprocess.run(["colab.exe", "new", "-s", session_name], check=True)
        print("   [OK] CPU Session READY.")

        print("3. Pushing mount runner to VM via stdin...")
        with open("mount_setup.py", "r", encoding="utf-8") as f:
            mount_code = f.read()

        runner_script = f"""
import json, subprocess, sys
with open('/content/mount_setup.py', 'w') as f:
    f.write({json.dumps(mount_code)})
subprocess.run([sys.executable, '/content/mount_setup.py', {json.dumps(client_id)}, {json.dumps(client_secret)}, {json.dumps(refresh_token)}], check=True)
"""
        cmd = ["colab.exe", "exec", "-s", session_name, "--timeout", "300"]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        stdout, _ = proc.communicate(input=runner_script)

        print("--- Remote Mount Output ---")
        print(stdout)
        print("---------------------------")
    finally:
        print(f"4. Releasing CPU session {session_name}...")
        subprocess.run(["colab.exe", "stop", "-s", session_name], check=False)
        print("   [OK] CPU Session terminated.")

if __name__ == "__main__":
    main()
