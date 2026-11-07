"""
extract_outfit_assets.py
Parse and curate AI costume assets from the local nvzhuang repository (D:/dev/active/nvzhuang).
Outputs:
1. Full indexed dataset: assets/outfits/all_outfits_index.json (2727 outfits)
2. Curated seasonal Lookbook recipes: assets/outfits/curated_lookbook_assets.json (for MOSS & FLUX pipelines)
"""

import os
import re
import json

NVZHUANG_DIR = "D:/dev/active/nvzhuang"
PROJECT_DIR = "D:/dev/active/minimax-h3-colab-pipeline"
OUT_DIR = os.path.join(PROJECT_DIR, "assets", "outfits")
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORY_MAP = {
    "SPR": {"zh": "春季穿搭", "en": "Spring Casual & Chic", "season": "spring"},
    "SUM": {"zh": "夏季清爽", "en": "Summer Light & Cool", "season": "summer"},
    "AUT": {"zh": "秋季层次", "en": "Autumn Layering & Knitwear", "season": "autumn"},
    "WIN": {"zh": "冬季保暖", "en": "Winter Coats & Warmth", "season": "winter"},
    "PTY": {"zh": "派对礼服", "en": "Party & Evening Dress", "season": "party"},
    "SWM": {"zh": "泳装海滩", "en": "Beach & Swimwear", "season": "swimwear"},
    "LNG": {"zh": "家居随性", "en": "Loungewear & Home", "season": "loungewear"},
    "SXY": {"zh": "个性风格", "en": "Chic Style & Cosplay", "season": "style"}
}

def parse_js_array(file_path, var_name):
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    pattern = rf"const\s+{var_name}\s*=\s*(\[.*?\]);"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find array {var_name} in {file_path}")
    raw_json = match.group(1)
    return json.loads(raw_json)

def clean_keywords_to_prompt(keywords):
    # Split keywords by space or comma, strip extra spaces, and construct natural prompt phrases
    words = [w.strip() for w in keywords.split() if w.strip()]
    return ", ".join(words)

def process_dataset():
    print(f"[INFO] Loading datasets from {NVZHUANG_DIR} ...")
    real_raw = parse_js_array(os.path.join(NVZHUANG_DIR, "data.js"), "D")
    anime_raw = parse_js_array(os.path.join(NVZHUANG_DIR, "data-anime.js"), "D_ANIME")

    all_outfits = []

    for item in real_raw:
        oid = item.get("id", "")
        prefix = oid.split("-")[0] if "-" in oid else "OTHER"
        cat_info = CATEGORY_MAP.get(prefix, {"zh": "其他", "en": "Other", "season": "other"})
        kw = item.get("キーワード", "") or item.get("keywords", "")
        thumb_rel = item.get("サムネ", "") or item.get("thumb", "")
        local_thumb = os.path.join(NVZHUANG_DIR, "assets", thumb_rel).replace("\\", "/")

        all_outfits.append({
            "id": oid,
            "type": "real",
            "category_prefix": prefix,
            "category_zh": cat_info["zh"],
            "category_en": cat_info["en"],
            "season": cat_info["season"],
            "keywords": kw,
            "prompt_component": clean_keywords_to_prompt(kw),
            "thumb_local": local_thumb if os.path.exists(local_thumb) else None,
            "thumb_rel": thumb_rel
        })

    for item in anime_raw:
        oid = item.get("id", "")
        prefix = oid.split("-")[0] if "-" in oid else "OTHER"
        cat_info = CATEGORY_MAP.get(prefix, {"zh": "其他", "en": "Other", "season": "other"})
        kw = item.get("キーワード", "") or item.get("keywords", "")
        thumb_rel = item.get("サムネ", "") or item.get("thumb", "")
        local_thumb = os.path.join(NVZHUANG_DIR, "assets", thumb_rel).replace("\\", "/")

        all_outfits.append({
            "id": oid,
            "type": "anime",
            "category_prefix": prefix,
            "category_zh": cat_info["zh"],
            "category_en": cat_info["en"],
            "season": cat_info["season"],
            "keywords": kw,
            "prompt_component": clean_keywords_to_prompt(kw),
            "thumb_local": local_thumb if os.path.exists(local_thumb) else None,
            "thumb_rel": thumb_rel
        })

    print(f"[OK] Total parsed outfits: {len(all_outfits)} (Real: {len(real_raw)}, Anime: {len(anime_raw)})")

    # Save full index
    full_index_path = os.path.join(OUT_DIR, "all_outfits_index.json")
    with open(full_index_path, "w", encoding="utf-8") as f:
        json.dump(all_outfits, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] Full index written to: {full_index_path}")

    # Build Curated 4-Season Lookbook recipes for MOSS (Real) and FLUX (Anime)
    curated = {
        "metadata": {
            "source": "https://github.com/paipiange/nvzhuang",
            "total_extracted": len(all_outfits),
            "purpose": "Decoupled Outfit Lookbook Generation for Moody Moss Mix & FLUX.1"
        },
        "real_photorealistic_lookbook": [
            {
                "id": "SPR-0002",
                "season": "Spring (春季)",
                "title": "法式优雅碎花吊带与针织开衫 (French Floral Camisole & Pleated Mini Skirt)",
                "outfit_prompt": "wearing yellow floral camisole cardigan, pink pleated mini skirt, ivory coat, delicate gold jewelry, elegant spring Parisian lookbook style",
                "reference_thumb": "D:/dev/active/nvzhuang/assets/thumbs/SPR-0002.webp"
            },
            {
                "id": "SUM-0005",
                "season": "Summer (夏季)",
                "title": "极简清爽亚麻衬衫与高腰长裤 (Minimalist Crisp Summer Linen & Trousers)",
                "outfit_prompt": "wearing crisp white sleeveless linen blouse, beige wide-leg tailored trousers, leather slide sandals, leather woven tote bag, airy sunlit summer aesthetic",
                "reference_thumb": "D:/dev/active/nvzhuang/assets/thumbs/SUM-0005.webp"
            },
            {
                "id": "AUT-0010",
                "season": "Autumn (秋季)",
                "title": "经典英伦风衣与燕麦色高领羊绒 (Classic Trench Coat & Oatmeal Cashmere Turtleneck)",
                "outfit_prompt": "wearing tailored camel trench coat, soft oatmeal knit turtleneck sweater, dark navy wool trousers, leather ankle boots, rich autumn foliage golden hour palette",
                "reference_thumb": "D:/dev/active/nvzhuang/assets/thumbs/AUT-0010.webp"
            },
            {
                "id": "WIN-0014",
                "season": "Winter (冬季)",
                "title": "重磅羊毛大衣与蓬松围巾质感 (Heavyweight Wool Overcoat & Fluffy Cashmere Scarf)",
                "outfit_prompt": "wearing oversized charcoal wool tailored overcoat, thick cream cable-knit scarf, ribbed leggings, leather combat boots, crisp winter snow-lit urban atmosphere",
                "reference_thumb": "D:/dev/active/nvzhuang/assets/thumbs/WIN-0014.webp"
            }
        ],
        "anime_stylized_lookbook": [
            {
                "id": "SPR-0017",
                "season": "Spring (春季学院)",
                "title": "春日元气水手领学院风 (Spring Sailor Collar Academy Uniform)",
                "outfit_prompt": "wearing navy sailor collar academy blouse, ivory pleated mini skirt, denim cropped jacket, knee-high socks, white sneakers, playful vibrant anime aesthetic",
                "reference_thumb": "D:/dev/active/nvzhuang/assets/anime/thumbs/SPR-0017.webp"
            },
            {
                "id": "SUM-0021",
                "season": "Summer (夏日海滨)",
                "title": "夏日阳光碎花轻盈连衣裙 (Sunlit Summer Floral Chiffon Sundress)",
                "outfit_prompt": "wearing sky blue chiffon floral sundress, delicate ribbons, straw sunhat, white buckled sandals, refreshing sunny seaside anime scenery, masterpiece",
                "reference_thumb": "D:/dev/active/nvzhuang/assets/anime/thumbs/SUM-0021.webp"
            },
            {
                "id": "AUT-0033",
                "season": "Autumn (秋日街头)",
                "title": "秋日复古机能夹克与百褶格裙 (Autumn Techwear Bomber & Plaid Pleated Skirt)",
                "outfit_prompt": "wearing olive green bomber jacket over oversized knit hoodie, beige tartan pleated skirt, chunky platform boots, stylish anime street fashion lookbook",
                "reference_thumb": "D:/dev/active/nvzhuang/assets/anime/thumbs/AUT-0033.webp"
            },
            {
                "id": "SXY-0137",
                "season": "Special (赛博朋克/风格化)",
                "title": "哥特机能束身短裙与战术长靴 (Gothic Cyber Corset Dress & Combat Boots)",
                "outfit_prompt": "wearing midnight navy corset mini dress, ruffled dark trim, tactical leather gloves, thigh-high stockings, platform boots, dramatic cyberpunk anime fashion",
                "reference_thumb": "D:/dev/active/nvzhuang/assets/anime/thumbs/SXY-0137.webp"
            }
        ]
    }

    curated_path = os.path.join(OUT_DIR, "curated_lookbook_assets.json")
    with open(curated_path, "w", encoding="utf-8") as f:
        json.dump(curated, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] Curated Lookbook recipes written to: {curated_path}")

if __name__ == "__main__":
    process_dataset()
