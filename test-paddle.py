import re
import cv2
import unicodedata
from PIL import Image

from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


IMAGE_PATH = "D:\OCR-Project\data\images\Screenshot 2026-03-10 214221.png"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def order_boxes(boxes):
    def key_fn(box):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return (min(ys), min(xs))
    return sorted(boxes, key=key_fn)


def crop_box(img, box, min_pad_x=6, min_pad_y=4):
    xs = [int(p[0]) for p in box]
    ys = [int(p[1]) for p in box]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    w = x2 - x1
    h = y2 - y1

    pad_x = max(min_pad_x, int(w * 0.03))
    pad_y = max(min_pad_y, int(h * 0.18))

    x1 = max(x1 - pad_x, 0)
    y1 = max(y1 - pad_y, 0)
    x2 = min(x2 + pad_x, img.shape[1])
    y2 = min(y2 + pad_y, img.shape[0])

    crop = img[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def valid_crop(crop):
    if crop is None or crop.size == 0:
        return False

    h, w = crop.shape[:2]
    if h < 16 or w < 25:
        return False
    if h > 220:
        return False
    return True


def preprocess_for_vietocr(crop):
    h, w = crop.shape[:2]
    scale = 2.5 if w < 900 else 1.8

    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return Image.fromarray(thresh)


def simple_score(text: str) -> float:
    if not text:
        return -1e9

    text = normalize_text(text)
    score = 0.0

    score += min(len(text), 120) * 0.05

    weird = len(re.findall(r"[^0-9A-Za-zÀ-ỹ\s,.;:!?()\"'/%+-]", text))
    score -= weird * 3.0

    if re.search(r"(.)\1\1\1", text):
        score -= 8.0

    return score


def load_vietocr(device="cpu"):
    config = Cfg.load_config_from_name("vgg_transformer")
    config["device"] = device
    config["predictor"]["beamsearch"] = False
    return Predictor(config)


def light_cleanup(text: str) -> str:
    text = normalize_text(text)

    replacements = {
        "(Ctrl)": "",
        "@CUP": "",
        "@cup": "",
    }

    for wrong, right in replacements.items():
        text = text.replace(wrong, right)

    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def group_lines(items, y_threshold=22):
    if not items:
        return []

    items = sorted(items, key=lambda x: (x["y"], x["x"]))

    grouped = []
    current = [items[0]]

    for item in items[1:]:
        prev = current[-1]
        if abs(item["y"] - prev["y"]) <= y_threshold:
            current.append(item)
        else:
            grouped.append(current)
            current = [item]

    grouped.append(current)

    merged_lines = []
    for group in grouped:
        group = sorted(group, key=lambda x: x["x"])
        line_text = " ".join(x["text"] for x in group if x["text"].strip())
        line_text = normalize_text(line_text)

        if line_text:
            merged_lines.append(line_text)

    return merged_lines


def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {IMAGE_PATH}")

    detector = PaddleOCR(lang="vi")
    recognizer = load_vietocr(device="cpu")

    det_results = detector.predict(img)

    if not det_results:
        return

    boxes = det_results[0].get("dt_polys", [])
    if boxes is None or len(boxes) == 0:
        return

    boxes = order_boxes(boxes)

    chosen_items = []

    for box in boxes:
        crop, rect = crop_box(img, box)

        if not valid_crop(crop):
            continue

        variants = []

        pil_1 = preprocess_for_vietocr(crop)
        text_1 = normalize_text(recognizer.predict(pil_1))
        variants.append(text_1)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        pil_2 = Image.fromarray(gray)
        text_2 = normalize_text(recognizer.predict(pil_2))
        variants.append(text_2)

        best_text = max(variants, key=simple_score)
        best_text = light_cleanup(best_text)

        if not best_text:
            continue

        x1, y1, _, _ = rect 

        chosen_items.append({
            "x": x1,
            "y": y1,
            "text": best_text
        })

    raw_lines = group_lines(chosen_items)
    raw_text = "\n".join(raw_lines)

    print(raw_text)


if __name__ == "__main__":
    main()