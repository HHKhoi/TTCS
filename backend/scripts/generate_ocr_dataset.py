import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# ===== CONFIG =====
OUTPUT_DIR = "dataset"
GT_FILE = "data.txt"

START_INDEX = 183
END_INDEX = 1000   # tạo đến 1000

FONT_DIR = "assets/fonts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== DATA =====
sentences = [
    "Artificial intelligence is transforming modern applications.",
    "Cloud computing enables flexible deployment strategies.",
    "Software systems are becoming increasingly complex and scalable.",
    "Data pipelines are essential for machine learning workflows.",
    "Education plays a vital role in shaping future generations.",
    "Online learning platforms have gained popularity.",
    "Students benefit from interactive and adaptive content.",
    "Research skills are critical in academic environments.",
    "Financial markets fluctuate based on various economic factors.",
    "Investment strategies require careful risk assessment.",
    "Digital banking has transformed user experiences.",
    "Cryptocurrency adoption is increasing worldwide.",
    "Healthcare systems rely on accurate data and diagnostics.",
    "Regular exercise contributes to overall well being.",
    "Medical research continues to evolve rapidly.",
    "Preventive care is essential for long term health."
]

# ===== LOAD FONTS =====
def load_fonts():
    fonts = []
    for f in os.listdir(FONT_DIR):
        if f.endswith(".ttf"):
            fonts.append(os.path.join(FONT_DIR, f))
    if not fonts:
        raise Exception("No fonts found in assets/fonts/")
    return fonts

fonts = load_fonts()

# ===== GENERATE TEXT =====
def generate_paragraph():
    return " ".join(random.choices(sentences, k=25))


# ===== WRAP TEXT =====
def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    current = ""

    for w in words:
        test = current + (" " if current else "") + w
        if draw.textlength(test, font=font) <= max_width:
            current = test
        else:
            lines.append(current)
            current = w

    if current:
        lines.append(current)

    return lines


# ===== CREATE IMAGE =====
def create_image(text, font_path, idx):
    font = ImageFont.truetype(font_path, random.randint(18, 24))

    width = random.randint(900, 1200)
    height = random.randint(400, 650)

    img = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    margin = 40
    max_width = width - 2 * margin

    lines = wrap_text(draw, text, font, max_width)

    # ===== chỉ lấy 8–10 dòng =====
    lines = lines[:random.randint(8, 10)]

    y = margin
    for line in lines:
        draw.text((margin, y), line, font=font, fill=(0, 0, 0))
        y += 30

    img = np.array(img)

    # blur nhẹ (optional)
    if random.random() < 0.2:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    path = os.path.join(OUTPUT_DIR, f"{idx}.png")
    cv2.imwrite(path, img)

    # ===== QUAN TRỌNG =====
    visible_text = " ".join(lines)

    return path, visible_text


# ===== MAIN =====
with open(GT_FILE, "a", encoding="utf-8") as f:
    for idx in range(START_INDEX, END_INDEX + 1):
        text = generate_paragraph()
        font = random.choice(fonts)

        path, gt = create_image(text, font, idx)

        # chuẩn hóa GT
        gt = " ".join(gt.split())

        f.write(f"{path}|{gt}\n")

        if idx % 50 == 0:
            print(f"Generated: {idx}")

print("DONE: Generated dataset up to 1000 images")