"""
build_combined_flux_tasks.py
Merges Anime 4-Season Lookbook with 4 specific outfit tests (SPR-0001, SPR-0010, SPR-0013, SPR-0044)
into a unified batch pipeline JSON for single-session zero-redundancy execution.
"""

import os
import json

WORKSPACE_DIR = "D:/dev/active/minimax-h3-colab-pipeline"

# Base character prompt & visual styling
CHAR_BASE = "A beautiful anime girl with twintail silver hair, expressive crystal violet eyes, gentle sweet smile"
STYLE_QUALITY = "masterpiece, Makoto Shinkai style, kanime, clean anime linework, luminous volumetric lighting, cinematic aesthetic, high quality 8k concept art"

# 1. Four-season Lookbook outfits (from curated assets)
season_outfits = [
    {
        "id": "SPR-0017",
        "season": "Spring (春季学院)",
        "title": "春日元气水手领学院风 (Spring Sailor Collar Academy)",
        "outfit_prompt": "wearing navy sailor collar academy blouse, ivory pleated mini skirt, denim cropped jacket, knee-high socks, white sneakers"
    },
    {
        "id": "SUM-0021",
        "season": "Summer (夏日海滨)",
        "title": "夏日阳光碎花轻盈连衣裙 (Sunlit Summer Floral Sundress)",
        "outfit_prompt": "wearing sky blue chiffon floral sundress, delicate ribbons, straw sunhat, white buckled sandals"
    },
    {
        "id": "AUT-0033",
        "season": "Autumn (秋日街头)",
        "title": "秋日复古机能夹克与百褶格裙 (Autumn Bomber & Tartan Skirt)",
        "outfit_prompt": "wearing olive green bomber jacket over oversized knit hoodie, beige tartan pleated skirt, chunky platform boots"
    },
    {
        "id": "SXY-0137",
        "season": "Special (赛博机能)",
        "title": "哥特机能束身短裙与战术长靴 (Gothic Cyber Corset Dress)",
        "outfit_prompt": "wearing midnight navy corset mini dress, ruffled dark trim, tactical leather gloves, thigh-high stockings, platform boots"
    }
]

# 2. Specific requested outfits
specific_outfits = [
    {
        "id": "SPR-0001",
        "season": "Special Test (特测 1)",
        "title": "荷叶边针织与丹宁短裙 (Ivory Frill Knit & Denim Mini Skirt)",
        "outfit_prompt": "wearing ivory frill knit sweater, light blue denim mini skirt, casual ivory coat, white socks and sneakers, charming relaxed spring outfit"
    },
    {
        "id": "SPR-0010",
        "season": "Special Test (特测 2)",
        "title": "黄色丝带衬衫裙与牛仔外套 (Yellow Ribbon Shirt Dress & Denim Jacket)",
        "outfit_prompt": "wearing bright yellow ribbon shirt dress, classic cropped denim jacket, ivory socks, clean white canvas sneakers, cheerful sunny aesthetic"
    },
    {
        "id": "SPR-0013",
        "season": "Special Test (特测 3)",
        "title": "天蓝蕾丝吊带与米色牛仔裤 (Blue Lace Camisole & Beige Denim Pants)",
        "outfit_prompt": "wearing pastel blue lace camisole, soft knit cardigan, beige tailored denim pants, leather pumps, delicate shoulder ribbon bag"
    },
    {
        "id": "SPR-0044",
        "season": "Special Test (特测 4)",
        "title": "薄荷绿雪纺吊带与百褶长裙 (Mint Green Ribbon Chiffon & Pleated Midi Skirt)",
        "outfit_prompt": "wearing mint green ribbon chiffon camisole, beige elegant pleated midi skirt, cropped denim jacket, stylish retro sandals"
    }
]

combined_tasks = []
all_specs = season_outfits + specific_outfits

for idx, item in enumerate(all_specs, 1):
    full_prompt = (
        f"{CHAR_BASE}, {item['outfit_prompt']}, "
        f"standing outdoors in front of picturesque Tokyo street with soft glowing evening sky and gentle lens flare, "
        f"full-body portrait, {STYLE_QUALITY}"
    )
    combined_tasks.append({
        "id": idx,
        "name": f"Flux_Anime_Lookbook_{idx}_{item['id']}",
        "outfit_id": item["id"],
        "season": item["season"],
        "title": item["title"],
        "lora_path": "anime/makoto_shinkai_flux.safetensors",
        "lora_strength": 0.85,
        "prompt": full_prompt,
        "width": 768,
        "height": 1024,
        "steps": 20,
        "guidance": 3.5,
        "seed": 2026
    })

output_file = os.path.join(WORKSPACE_DIR, "flux_combined_tasks.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(combined_tasks, f, indent=2, ensure_ascii=False)

print(f"[OK] Generated {len(combined_tasks)} combined tasks in: {output_file}")
