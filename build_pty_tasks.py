"""
build_pty_tasks.py
Generates 48 full-body evening dress & party photoshoot tasks for model f71f980dde3.
Strictly guarantees full-body framing (head-to-toe, footwear visible, ample ground space).
"""

import os
import json

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_IMAGE_PATH = "D:/dev/active/nvzhuang/assets/portrait/thumbs/f71f980dde3.webp"

# Model f71f visual identity anchor
CHAR_PROMPT = (
    "The exact same young Japanese fashion model from reference image, "
    "cheerful sweet smile, sparkling amber-brown eyes with delicate aegyo-sal, "
    "neat chin-length dark brown bob haircut with airy see-through bangs, "
    "glowing porcelain skin, natural elegant evening makeup"
)

# Explicit Full-Body Framing specification
FULL_BODY_FRAMING = (
    "full-length body fashion photograph, complete head-to-toe shot, "
    "entire figure captured from head down to shoes, high heels touching the polished floor, "
    "ample ground space visible beneath feet, generous headroom above, "
    "editorial fashion lookbook catalog full body standing pose, wide angle portrait"
)

# Photo quality specification
PHOTO_QUALITY = (
    "masterpiece, ultra-realistic photograph, Hasselblad 80mm lens, "
    "cinematic soft atmospheric lighting, natural shadows on floor, "
    "photorealistic skin texture, crisp fabric sheen, 8k resolution, raw photo"
)

# Sceneries for Party / Evening Dress (PTY)
SCENES = [
    "standing elegantly on a grand marble ballroom floor with distant glowing crystal chandeliers",
    "standing gracefully on a red carpet gala entrance with warm ambient flash lights",
    "standing by the illuminated glass balustrade of a luxury rooftop cocktail lounge at twilight",
    "standing poised in a high-end luxury hotel lobby with warm golden architectural lighting",
    "standing near an opulent champagne bar with subtle candlelight reflections on marble floor",
    "standing beside a grand spiral staircase in a historic European opera hall"
]

def build_outfit_description(item):
    kw = item.get("keywords", "").lower()
    
    # Expand shorthand keywords into refined fashion prose
    elements = []
    
    # Dress/Skirt
    if "mini dress" in kw:
        color = next((c for c in ["red", "black", "blue", "mint", "pink", "ivory", "white", "beige", "navy", "lavender"] if c in kw), "")
        fabric = next((f for f in ["velvet", "satin", "chiffon", "lace", "tulle"] if f in kw), "")
        dress_str = f"{color} {fabric} mini dress".strip()
        if "camisole" in kw:
            dress_str = f"camisole {dress_str}"
        elements.append(f"wearing tailored {dress_str}")
    elif "midi dress" in kw:
        color = next((c for c in ["red", "black", "blue", "mint", "pink", "ivory", "white", "beige", "navy", "lavender"] if c in kw), "")
        fabric = next((f for f in ["velvet", "satin", "chiffon", "lace", "tulle"] if f in kw), "")
        elements.append(f"wearing elegant {color} {fabric} midi dress".strip())
    elif "skirt" in kw:
        color = next((c for c in ["red", "black", "blue", "mint", "pink", "ivory", "white", "beige", "navy", "lavender"] if c in kw), "")
        fabric = next((f for f in ["velvet", "satin", "chiffon", "lace", "tulle"] if f in kw), "")
        elements.append(f"wearing chic {color} {fabric} skirt paired with lace camisole top".strip())
    else:
        elements.append(f"wearing sophisticated evening attire featuring {kw}")

    # Tulle / Ribbon / Fur accents
    if "tulle skirt" in kw and "tulle" not in elements[0]:
        elements.append("fluffy layered tulle skirt overlay")
    if "ribbon" in kw:
        elements.append("dainty satin ribbon accent")
    if "fur" in kw:
        elements.append("soft faux-fur luxury evening stole")
        
    # Gloves
    if "gloves" in kw:
        glove_color = next((c for c in ["black", "mint", "pink", "blue", "beige", "white"] if c in kw), "matching")
        elements.append(f"opera-length {glove_color} satin evening gloves")
        
    # Hosiery
    if "tights" in kw:
        if "black" in kw:
            elements.append("sheer black evening tights")
        else:
            elements.append("sheer elegant tights")
    else:
        elements.append("slender bare legs")
        
    # Footwear (Guarantees Full Body Foot Rendering)
    if "pumps" in kw:
        elements.append("matching designer pointed-toe high heel pumps")
    elif "heels" in kw:
        elements.append("glossy stiletto high heels")
    else:
        elements.append("elegant evening high heels")
        
    # Accessories
    accs = []
    if "necklace" in kw:
        accs.append("sparkling diamond pendant necklace")
    if "mini bag" in kw or "bag" in kw:
        accs.append("metallic clutch mini bag")
    if accs:
        elements.append(", ".join(accs))
        
    return ", ".join(elements)

def main():
    with open(os.path.join(WORKSPACE_DIR, "assets", "outfits", "all_outfits_index.json"), "r", encoding="utf-8") as f:
        all_data = json.load(f)
        
    pty_list = [x for x in all_data if x.get("type") == "real" and x.get("category_prefix") == "PTY"]
    pty_list.sort(key=lambda x: x["id"])
    
    print(f"Loaded {len(pty_list)} real PTY outfits.")
    
    tasks = []
    for idx, item in enumerate(pty_list, 1):
        outfit_desc = build_outfit_description(item)
        scene = SCENES[(idx - 1) % len(SCENES)]
        
        # Assemble complete prompt with explicit full-body framing guarantees
        full_prompt = f"{FULL_BODY_FRAMING}, {CHAR_PROMPT}, {outfit_desc}, {scene}, {PHOTO_QUALITY}"
        
        task = {
            "id": idx,
            "name": f"PTY_FullBody_{idx:02d}_{item['id']}",
            "outfit_id": item["id"],
            "category": "PTY (晚礼服与派对高定)",
            "keywords": item.get("keywords", ""),
            "ref_image": REF_IMAGE_PATH,
            "prompt": full_prompt,
            "negative_prompt": "cropped feet, cut off shoes, upper body only, waist up, close up, out of frame legs, bad legs, extra limbs, deformed toes, floating feet, blurry floor, low resolution",
            "neg_prompt": "cropped feet, cut off shoes, upper body only, waist up, close up, out of frame legs, bad legs, extra limbs, deformed toes, floating feet, blurry floor, low resolution",
            "width": 896,
            "height": 1344,
            "steps": 25,
            "guidance": 3.5,
            "seed": 10000 + idx * 77
        }
        tasks.append(task)
        
    output_path = os.path.join(WORKSPACE_DIR, "photoshoot_pty_tasks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
        
    print(f"[OK] Successfully built {len(tasks)} PTY photoshoot tasks -> {output_path}")
    print("\nSample Task Prompt Preview (Task 1):")
    print(tasks[0]["prompt"])

if __name__ == "__main__":
    main()
