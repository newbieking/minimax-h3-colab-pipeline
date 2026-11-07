"""
generate_lookbook_tasks.py
Generates decoupled Lookbook batch tasks for:
1. Moody Moss Mix (Photorealistic 4-Season Lookbook) -> moss_lookbook_tasks.json
2. FLUX.1 [dev] (Anime 4-Season Lookbook) -> flux_lookbook_tasks.json
"""

import os
import json

WORKSPACE_DIR = "D:/dev/active/minimax-h3-colab-pipeline"
ASSETS_FILE = os.path.join(WORKSPACE_DIR, "assets", "outfits", "curated_lookbook_assets.json")

with open(ASSETS_FILE, "r", encoding="utf-8") as f:
    curated = json.load(f)

# ==============================================================================
# 1. MOODY MOSS MIX (Photorealistic 4-Season Lookbook)
# Locked character identity: "A stunning 24-year-old East Asian fashion model with natural clean makeup, delicate facial features, gentle gaze, styled soft wavy dark hair"
# Fixed Seed: 8888, Fixed Steps: 25, CFG: 3.0, Resolution: 896x1344 (portrait)
# ==============================================================================
moss_character_base = "A stunning 24-year-old East Asian fashion model with natural clean makeup, delicate facial features, gentle gaze, styled soft wavy dark hair"
moss_photo_quality = "ultra-realistic, Hasselblad 80mm f/1.8 lens portrait, natural daylight soft lighting, film grain, subsurface skin scattering, cinematic fashion magazine editorial, masterpiece, 8k resolution"

moss_tasks = []
for idx, outfit in enumerate(curated["real_photorealistic_lookbook"], 1):
    prompt = f"{moss_character_base}, {outfit['outfit_prompt']}, standing in a minimalist modern architectural studio with neutral background, {moss_photo_quality}"
    moss_tasks.append({
        "id": idx,
        "name": f"Moss_Lookbook_{idx}_{outfit['id']}",
        "season": outfit["season"],
        "outfit_id": outfit["id"],
        "title": outfit["title"],
        "prompt": prompt,
        "width": 896,
        "height": 1344,
        "steps": 25,
        "cfg": 3.0,
        "seed": 8888,
        "count": 1
    })

moss_tasks_path = os.path.join(WORKSPACE_DIR, "moss_lookbook_tasks.json")
with open(moss_tasks_path, "w", encoding="utf-8") as f:
    json.dump(moss_tasks, f, indent=2, ensure_ascii=False)
print(f"[OK] Generated {len(moss_tasks)} Moody Moss Lookbook tasks -> {moss_tasks_path}")

# ==============================================================================
# 2. FLUX.1 [dev] (Anime 4-Season Lookbook)
# Locked character identity: "A beautiful anime girl with twintail silver hair, expressive crystal violet eyes, slight smile"
# Style LoRA: anime/makoto_shinkai_flux.safetensors or anime_crabdm_flux
# Fixed Seed: 2026, Fixed Steps: 20, Guidance: 3.5, Resolution: 768x1024
# ==============================================================================
flux_character_base = "A beautiful anime girl with twintail silver hair, expressive crystal violet eyes, gentle sweet smile"
flux_quality = "masterpiece, Makoto Shinkai style, kanime, clean linework, vibrant aesthetic, high quality anime character concept art"

flux_tasks = []
for idx, outfit in enumerate(curated["anime_stylized_lookbook"], 1):
    prompt = f"{flux_character_base}, {outfit['outfit_prompt']}, standing in front of soft aesthetic watercolor gradient background, full body view, {flux_quality}"
    flux_tasks.append({
        "id": idx,
        "name": f"Flux_Anime_Lookbook_{idx}_{outfit['id']}",
        "season": outfit["season"],
        "outfit_id": outfit["id"],
        "title": outfit["title"],
        "lora_path": "anime/makoto_shinkai_flux.safetensors",
        "lora_strength": 0.85,
        "prompt": prompt,
        "width": 768,
        "height": 1024,
        "steps": 20,
        "guidance": 3.5,
        "seed": 2026
    })

flux_tasks_path = os.path.join(WORKSPACE_DIR, "flux_lookbook_tasks.json")
with open(flux_tasks_path, "w", encoding="utf-8") as f:
    json.dump(flux_tasks, f, indent=2, ensure_ascii=False)
print(f"[OK] Generated {len(flux_tasks)} FLUX.1 Anime Lookbook tasks -> {flux_tasks_path}")
