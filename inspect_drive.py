import subprocess
import json

def gws_drive_list(q=None, page_size=50):
    params = {"pageSize": page_size}
    if q:
        params["q"] = q
    res = subprocess.run(["gws.cmd", "drive", "files", "list", "--params", json.dumps(params)], capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        print("Error:", res.stderr)
        return None
    return json.loads(res.stdout)

if __name__ == "__main__":
    print("--- Listing files inside MiniMax_Models ---")
    data = gws_drive_list(q="'1DnNwDtxblkLra2cVtqNTlaqPBXIHByPB' in parents")
    if data:
        for f in data.get("files", []):
            print(f"{f['id']} | {f['mimeType']} | {f['name']}")

    print("\n--- Listing subfolders in MiniMax_Models ---")
    # Also let's inspect vae, diffusion_models, text_encoders, loras if any
    for sub in ["diffusion_models", "text_encoders", "loras", "vae"]:
        sub_data = gws_drive_list(q=f"name = '{sub}' and mimeType = 'application/vnd.google-apps.folder'")
        if sub_data and sub_data.get("files"):
            folder_id = sub_data["files"][0]["id"]
            print(f"\nFolder: {sub} ({folder_id})")
            items = gws_drive_list(q=f"'{folder_id}' in parents")
            for item in items.get("files", []):
                print(f"  {item['id']} | {item['name']}")

print("\n--- Listing MiniMax_H3_Outputs ---")
out_data = gws_drive_list(q="'1NQTHdEb0gkuwuOt7SnNEWvnwnfQkcdfk' in parents")
if out_data:
    for f in out_data.get("files", []):
        print(f"{f['id']} | {f['name']} | {f.get('size')} bytes")
