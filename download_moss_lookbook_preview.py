import os
import sys
import json
import subprocess

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORKSPACE_DIR)

GWS_BIN = "gws.cmd" if sys.platform == "win32" else "gws"

def download():
    local_dir = "moss_lookbook_preview"
    os.makedirs(local_dir, exist_ok=True)

    search_params = json.dumps({
        "q": "name = 'Moody_Moss_Lookbook' and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
        "fields": "files(id, name)"
    })
    res = subprocess.check_output([GWS_BIN, "drive", "files", "list", "--params", search_params], text=True, encoding="utf-8")
    data = json.loads(res)
    folders = data.get("files", [])
    if not folders:
        print("[ERR] Folder Moody_Moss_Lookbook not found!")
        return

    folder_id = folders[0]["id"]
    print(f"[OK] Found Moody_Moss_Lookbook folder: {folder_id}")

    file_params = json.dumps({
        "q": f"'{folder_id}' in parents and trashed = false",
        "fields": "files(id, name, mimeType)"
    })
    res_files = subprocess.check_output([GWS_BIN, "drive", "files", "list", "--params", file_params], text=True, encoding="utf-8")
    files_data = json.loads(res_files)

    for item in files_data.get("files", []):
        file_id = item["id"]
        file_name = item["name"]
        # Use relative path without leading slash
        dest_rel = f"{local_dir}/{file_name}"
        print(f"Downloading {file_name} -> {dest_rel} ...")
        params = json.dumps({"fileId": file_id, "alt": "media"})
        cmd = [GWS_BIN, "drive", "files", "get", "--params", params, "-o", dest_rel]
        subprocess.run(cmd, check=True)

    print("[SUCCESS] All Moody Moss Lookbook files downloaded locally!")

if __name__ == "__main__":
    download()
