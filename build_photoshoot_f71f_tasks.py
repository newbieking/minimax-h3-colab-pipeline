"""
build_photoshoot_f71f_tasks.py
Builds the 8-lookbook photoshoot tasks for model f71f980dde3 with reference image conditioning.
8 Outfits sampled across: SPR, SUM, AUT, WIN, PTY, SWM, LNG, SXY.
"""

import os
import json

WORKSPACE_DIR = "D:/dev/active/minimax-h3-colab-pipeline"
REF_IMAGE_PATH = "D:/dev/active/nvzhuang/assets/portrait/thumbs/f71f980dde3.webp"

# Model f71f980dde3 prompt anchor
CHAR_PROMPT = "The exact same young Japanese fashion model from reference image, cheerful sweet smile, sparkling amber-brown eyes with delicate aegyo-sal, neat chin-length dark brown bob haircut with airy see-through bangs, glowing porcelain skin, natural nude makeup"

PHOTO_QUALITY = "masterpiece, ultra-realistic photograph, Hasselblad 80mm lens, natural daylight soft lighting, film grain, full-body portrait, cinematic aesthetic, 8k resolution"

outfits = [
    {
        "id": "SPR-0035",
        "category": "Spring (春季日常)",
        "title": "明黄碎花法式中长裙 (Yellow Floral Midi Dress & Beige Tights)",
        "outfit_prompt": "wearing yellow floral midi dress, sheer beige tights, strappy sandals, mini tote bag, delicate gold necklace",
        "scene_prompt": "standing gracefully in a sunlit Tokyo street corner cafe"
    },
    {
        "id": "SUM-0244",
        "category": "Summer (夏日清爽)",
        "title": "黑白复古波点吊带裙 (Polka-Dot Camisole Midi Dress)",
        "outfit_prompt": "wearing black and white polka-dot camisole midi dress, leather slide sandals, woven tote bag, silver pendant necklace",
        "scene_prompt": "standing by modern art gallery corridor with soft sunbeams through blinds"
    },
    {
        "id": "AUT-0131",
        "category": "Autumn (金秋格调)",
        "title": "象牙白针织与橄榄绿阔腿裤 (Ivory Knit & Olive Green Trousers)",
        "outfit_prompt": "wearing soft ivory ribbed knit sweater, high-waisted olive green tailored trousers, leather loafers, leather shoulder bag",
        "scene_prompt": "standing outdoors on a golden hour tree-lined boulevard"
    },
    {
        "id": "WIN-0250",
        "category": "Winter (冬日奢华)",
        "title": "皮草外套与修身黑色针织短裙 (Fur Coat & Black Knit Mini Dress)",
        "outfit_prompt": "wearing plush oversized faux fur coat over black knit bodycon mini dress, dark tights, knee-high leather boots",
        "scene_prompt": "standing in front of illuminated holiday city shopfronts with soft falling snow"
    },
    {
        "id": "PTY-0038",
        "category": "Party (派对晚宴)",
        "title": "酒红丝绒吊带抹胸礼服 (Red Camisole Velvet Mini Dress)",
        "outfit_prompt": "wearing deep red velvet camisole cocktail mini dress, sheer black tights, elegant stiletto heels, delicate diamond necklace",
        "scene_prompt": "standing near an upscale cocktail lounge bar with warm ambient candlelights"
    },
    {
        "id": "SWM-0043",
        "category": "Swimwear (海滩度假)",
        "title": "粉色针织分体裙与草帽 (Pink Knit Swimwear & Beach Skirt)",
        "outfit_prompt": "wearing pastel pink knit two-piece swimwear with matching lightweight beach mini skirt, woven straw sunhat, beach tote",
        "scene_prompt": "standing on white sandy tropical beach with crystal turquoise waters"
    },
    {
        "id": "LNG-0133",
        "category": "Loungewear (居家私房)",
        "title": "墨绿蕾丝复古居家吊带裙 (Green Lace Camisole Loungewear)",
        "outfit_prompt": "wearing emerald green silk and lace camisole slip dress, sheer lightweight robe, delicate silver necklace",
        "scene_prompt": "standing in a warm sunlit minimalist modern apartment bedroom"
    },
    {
        "id": "SXY-0069",
        "category": "Chic Style (高定时装)",
        "title": "粉黑拼色束胸短裙与晚宴礼帽 (Pink Black Corset Mini Dress & Hat)",
        "outfit_prompt": "wearing designer pink and black corset structured mini dress, ruffled dark trim, black evening gloves, stylish cocktail hat",
        "scene_prompt": "standing against neutral minimalist concrete studio backdrop with high fashion lighting"
    }
]

tasks = []
for idx, o in enumerate(outfits, 1):
    prompt = f"{CHAR_PROMPT}, {o['outfit_prompt']}, {o['scene_prompt']}, {PHOTO_QUALITY}"
    tasks.append({
        "id": idx,
        "name": f"Photoshoot_f71f_{idx}_{o['id']}",
        "outfit_id": o["id"],
        "category": o["category"],
        "title": o["title"],
        "ref_image": REF_IMAGE_PATH,
        "prompt": prompt,
        "width": 896,
        "height": 1344,
        "steps": 25,
        "guidance": 3.5,
        "seed": 9999
    })

tasks_file = os.path.join(WORKSPACE_DIR, "photoshoot_f71f_tasks.json")
with open(tasks_file, "w", encoding="utf-8") as f:
    json.dump(tasks, f, indent=2, ensure_ascii=False)

print(f"[OK] Generated {len(tasks)} image-guided photoshoot tasks -> {tasks_file}")
