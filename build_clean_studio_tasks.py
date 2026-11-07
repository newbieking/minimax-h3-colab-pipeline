"""
build_clean_studio_tasks.py
Builds 10 clean-studio / solid-backdrop fashion catalog tasks for model f71f980dde3.
Focus:
- 100% clean, tidy, solid-color seamless cyclorama studio backdrop (zero clutter/furniture).
- Full-body head-to-toe framing with grounded footwear and contact shadows.
- Emphasis on garment texture, drapery, and dynamic model posture.
"""

import os
import json

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_IMAGE_PATH = "D:/dev/active/nvzhuang/assets/portrait/thumbs/f71f980dde3.webp"

# Model f71f identity anchor
CHAR_PROMPT = (
    "The exact same young Japanese fashion model from reference image, "
    "cheerful confident smile, sparkling amber-brown eyes with delicate aegyo-sal, "
    "neat chin-length dark brown bob haircut with airy see-through bangs, "
    "glowing porcelain skin, natural high-fashion makeup"
)

# Clean, solid-color, uncluttered studio backdrop
CLEAN_STUDIO_BACKDROP = (
    "standing in a pristine clean minimalist photography studio, "
    "seamless solid light warm-grey cyclorama backdrop, immaculate neutral solid floor, "
    "subtle soft contact shadows beneath shoes, zero clutter, zero furniture, zero decorations, "
    "completely clean tidy seamless solid color background"
)

# Full body head-to-toe guarantee
FULL_BODY_FRAMING = (
    "extreme full-length body fashion photograph, head-to-toe shot, "
    "entire figure captured from head down to shoes, feet firmly touching the clean floor, "
    "ample headroom above and foot room below, wide catalog framing, professional lookbook composition"
)

# Studio lighting and photographic texture
PHOTO_QUALITY = (
    "masterpiece, commercial fashion lookbook photography, large octabox soft diffused lighting, "
    "even balanced exposure, Hasselblad 80mm lens, tack-sharp focus across entire body and garments, "
    "rich fabric weave and material texture, 8k resolution, raw photo"
)

# Negative prompt strictly forbidding clutter and cropped framing
NEGATIVE_PROMPT = (
    "clutter, complex background, room interior, furniture, chandelier, outdoor, street, trees, "
    "props, posters, blurry floor, cropped feet, shoes cut off, waist up, close up, head out of frame, "
    "bad anatomy, extra limbs, deformed hands, floating feet, oversaturated"
)

# 10 curated outfits with tailored poses highlighting the garment
OUTFIT_SPECS = [
    {
        "id": "SPR-0080",
        "category": "Spring (春季叠穿)",
        "title": "米色蕾丝开衫与蓝褶短裙 (Beige Cardigan & Blue Pleated Skirt)",
        "outfit": "wearing delicate beige lace camisole under ivory knit cardigan, navy blue pleated mini skirt, tailored ivory lightweight coat, sheer stockings, beige leather loafer heels",
        "pose": "standing in a graceful model runway pose with one hand in coat pocket, weight subtly shifted to one hip, showcasing coat silhouette and skirt pleats"
    },
    {
        "id": "SUM-0249",
        "category": "Summer (都市极简)",
        "title": "黑白衬衫与高腰微喇牛仔裤 (Black & White Shirt & Denim Jeans)",
        "outfit": "wearing crisp tailored black and white color-block button-up shirt tucked into high-waisted blue denim flared jeans, leather slide sandals, leather belt, silver choker necklace",
        "pose": "confident editorial stance with both hands resting lightly in denim pockets, relaxed upright posture, showcasing the long leg line and denim drapery"
    },
    {
        "id": "AUT-0083",
        "category": "Autumn (金秋红韵)",
        "title": "酒红百褶长裙与皮靴 (Wine Red Pleated Midi Skirt & Leather Boots)",
        "outfit": "wearing soft cream ribbed knit sweater, high-waisted wine red pleated flowing midi skirt, knee-high cognac brown leather boots, delicate gold chain necklace, leather handbag",
        "pose": "poised three-quarter turn posture with one foot stepped forward, skirt fabric swishing softly, highlighting boot leather texture and pleated flow"
    },
    {
        "id": "WIN-0085",
        "category": "Winter (冬日复古)",
        "title": "红丝绒抹胸与毛呢大衣 (Red Velvet Top & Fur Trimmed Coat)",
        "outfit": "wearing deep red velvet camisole top, tailored chocolate brown wide-leg wool trousers, oversized cream coat with faux-fur collar, black leather ankle boots, ribbon belt",
        "pose": "fashion catalog standing pose with hands gently resting at hips, coat draped over shoulders, highlighting tailored trouser drape and velvet sheen"
    },
    {
        "id": "PTY-0020",
        "category": "Party (高定礼裙)",
        "title": "象牙白缎面吊带礼服短裙 (Ivory Satin Camisole Dress)",
        "outfit": "wearing luxurious ivory silk satin camisole mini dress with subtle bias cut, sheer pearl tights, metallic silver stiletto high heels, diamond pendant",
        "pose": "glamorous full-body standing pose with one leg slightly crossed in front, arms relaxed at sides, highlighting fluid silk drapery and stiletto heels"
    },
    {
        "id": "SWM-0019",
        "category": "Swimwear (极简剪裁)",
        "title": "黑金经典连体泳装 (Black & Gold Minimalist Swimsuit)",
        "outfit": "wearing sculptural matte black one-piece swimsuit with metallic gold belt buckle accent, sleek slide sandals, oversized designer sunglasses held in hand",
        "pose": "tall athletic model posture with feet planted firmly shoulder-width apart, arms naturally poised, emphasizing clean body lines and architectural cutouts"
    },
    {
        "id": "LNG-0176",
        "category": "Loungewear (居家丝缎)",
        "title": "绯红蕾丝真丝居家睡裙 (Crimson Silk & Lace Loungewear)",
        "outfit": "wearing crimson red silk slip dress with delicate scalloped black lace trim, matching lightweight open silk robe, sheer stockings, minimalist satin indoor slippers",
        "pose": "relaxed graceful standing posture with robe flowing gently, hands poised near waist, showcasing the airy lightness of the silk and lace hem"
    },
    {
        "id": "SXY-0054",
        "category": "Chic High-Fashion (法式高定)",
        "title": "卡其拼色围裙风束胸裙 (Beige & White Frill Corset Dress)",
        "outfit": "wearing couture beige structured corset mini dress with white ruffled frill bib trim, delicate white lace patterned stockings, pointed ivory pump heels",
        "pose": "couture high-fashion pose standing straight with chin subtly raised, one hand touching collarbone, showcasing corset structure and stockings"
    },
    {
        "id": "SPR-0011",
        "category": "Spring (清新风尚)",
        "title": "鼠尾草绿短裙与风衣 (Sage Green Mini Skirt & Trench Coat)",
        "outfit": "wearing sage green tailored high-waisted shorts-skirt, white fitted top, classic double-breasted ivory trench coat, white crew socks, chunky strap platform sandals, necklace",
        "pose": "playful modern street-chic pose walking forward mid-stride, showing movement of coat and sandals firmly contacting the floor"
    },
    {
        "id": "WIN-0016",
        "category": "Winter (轻奢名媛)",
        "title": "烟灰高领毛衣与毛绒短裙 (Grey Turtleneck Knit & Fur Mini Skirt)",
        "outfit": "wearing fine-gauge grey turtleneck ribbed sweater, black plush faux-fur mini skirt, metallic gold trim bomber jacket, sheer tights, patent leather chelsea boots",
        "pose": "confident chic winter pose with one hand in jacket pocket, head tilted with a warm radiant smile, showcasing knit weave and fur texture"
    }
]

def main():
    tasks = []
    for idx, spec in enumerate(OUTFIT_SPECS, 1):
        prompt = (
            f"{FULL_BODY_FRAMING}, {CHAR_PROMPT}, {spec['outfit']}, "
            f"{spec['pose']}, {CLEAN_STUDIO_BACKDROP}, {PHOTO_QUALITY}"
        )
        task = {
            "id": idx,
            "name": f"CleanStudio_FullBody_{idx:02d}_{spec['id']}",
            "outfit_id": spec["id"],
            "category": spec["category"],
            "title": spec["title"],
            "ref_image": REF_IMAGE_PATH,
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "neg_prompt": NEGATIVE_PROMPT,
            "width": 896,
            "height": 1344,
            "steps": 25,
            "guidance": 3.5,
            "seed": 20000 + idx * 83
        }
        tasks.append(task)

    output_path = os.path.join(WORKSPACE_DIR, "clean_studio_10_tasks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"[OK] Successfully built {len(tasks)} Clean Studio catalog tasks -> {output_path}")
    print("\nSample Task Prompt Preview (Task 1):")
    print(tasks[0]["prompt"])

if __name__ == "__main__":
    main()
