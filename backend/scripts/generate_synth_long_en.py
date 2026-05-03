import os
import random
import shutil
import unicodedata
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "train_rec" / "data" / "rec_long_en"
TRAIN_DIR = OUT_DIR / "train"
VAL_DIR = OUT_DIR / "val"
TRAIN_GT = OUT_DIR / "rec_gt_train.txt"
VAL_GT = OUT_DIR / "rec_gt_val.txt"
CORPUS_FILE = BASE_DIR / "train_rec" / "corpus" / "lines.txt"
NUM_SAMPLES = 12000
TRAIN_RATIO = 0.9
IMG_HEIGHT = 48
SEED = 42

FONT_PATHS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/georgia.ttf",
    "C:/Windows/Fonts/cambria.ttc",
]

def load_corpus():
    if not CORPUS_FILE.exists():
        raise FileNotFoundError(f"Missing corpus file: {CORPUS_FILE}")
    lines = []
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            text = " ".join(line.strip().split())
            text = unicodedata.normalize("NFC", text)
            if text:
                lines.append(text)
    if not lines:
        raise ValueError("Corpus is empty.")
    return lines

def get_font():
    valid_fonts = [f for f in FONT_PATHS if os.path.exists(f)]
    if not valid_fonts:
        raise FileNotFoundError("No valid fonts found.")
    font_path = random.choice(valid_fonts)
    font_size = random.randint(22, 34)
    return ImageFont.truetype(font_path, font_size)

def normalize_quotes_and_dashes(text):
    replacements = [
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
        ("—", "-"),
        ("–", "-"),
    ]
    for a, b in replacements:
        if random.random() < 0.35:
            text = text.replace(a, b)
    return text

def maybe_mutate_text(text):
    text = unicodedata.normalize("NFC", text)

    if random.random() < 0.35:
        text = normalize_quotes_and_dashes(text)

    if random.random() < 0.20:
        text = text.replace("  ", " ")

    if random.random() < 0.15:
        text = text.replace(" ,", ",").replace(" .", ".")

    return text

def render_text_image(text):
    font = get_font()

    dummy = Image.new("RGB", (10, 10), (255, 255, 255))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_x = random.randint(16, 28)
    pad_y = random.randint(6, 10)

    width = max(240, min(1800, text_w + pad_x * 2))
    height = max(IMG_HEIGHT, text_h + pad_y * 2)

    bg_val = random.randint(242, 255)
    fg_val = random.randint(0, 60)

    img = Image.new("RGB", (width, height), (bg_val, bg_val, bg_val))
    draw = ImageDraw.Draw(img)

    x = random.randint(8, 18)
    y = max(0, (height - text_h) // 2 - bbox[1])

    draw.text((x, y), text, font=font, fill=(fg_val, fg_val, fg_val))

    if random.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.3))

    if random.random() < 0.08:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6))

    return img

def reset_output_dir():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    random.seed(SEED)
    corpus = load_corpus()

    reset_output_dir()

    with open(TRAIN_GT, "w", encoding="utf-8") as f_train, \
         open(VAL_GT, "w", encoding="utf-8") as f_val:

        for i in range(NUM_SAMPLES):
            text = maybe_mutate_text(random.choice(corpus))
            img = render_text_image(text)

            file_name = f"{i:06d}.png"

            if random.random() < TRAIN_RATIO:
                img_path = TRAIN_DIR / file_name
                img.save(img_path)
                f_train.write(f"{img_path.as_posix()}\t{text}\n")
            else:
                img_path = VAL_DIR / file_name
                img.save(img_path)
                f_val.write(f"{img_path.as_posix()}\t{text}\n")

            if (i + 1) % 1000 == 0:
                print(f"[OK] {i + 1}/{NUM_SAMPLES}")

    print("[DONE] Generated English punctuation-aware dataset")

if __name__ == "__main__":
    main()