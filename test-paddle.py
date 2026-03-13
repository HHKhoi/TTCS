import re
import cv2
import unicodedata
from PIL import Image

from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


IMAGE_PATH = "./data/images/Screenshot 2026-03-10 214221.png"


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


def load_vietnamese_corrector():
    model_name = "bmd1905/vietnamese-correction"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def split_into_chunks(text, max_len=220):
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    chunks = []
    cur = ""

    for line in lines:
        if len(cur) + len(line) + 1 <= max_len:
            cur = (cur + " " + line).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = line

    if cur:
        chunks.append(cur)

    return chunks


def correct_text_with_lm(text, tokenizer, model):
    chunks = split_into_chunks(text)
    corrected_chunks = []

    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True
        )

        corrected = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        corrected_chunks.append(corrected)

    return "\n".join(corrected_chunks)


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

    print("Dang load PaddleOCR...")
    detector = PaddleOCR(
        lang="vi",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    print("Dang load VietOCR...")
    recognizer = load_vietocr(device="cpu")

    print("Dang detect boxes...")
    det_results = detector.predict(img)

    if not det_results:
        print("Khong co ket qua detect.")
        return

    boxes = det_results[0].get("dt_polys", [])
    if boxes is None or len(boxes) == 0:
        print("Khong detect duoc box nao.")
        return

    boxes = order_boxes(boxes)
    print("So box detect duoc:", len(boxes))

    debug_img = img.copy()
    chosen_items = []

    for i, box in enumerate(boxes):
        crop, rect = crop_box(img, box)

        if not valid_crop(crop):
            continue

        # thử 2 biến thể đơn giản
        variants = []

        pil_1 = preprocess_for_vietocr(crop)
        text_1 = normalize_text(recognizer.predict(pil_1))
        variants.append(("adaptive", text_1))

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        pil_2 = Image.fromarray(gray)
        text_2 = normalize_text(recognizer.predict(pil_2))
        variants.append(("gray", text_2))

        best_variant, best_text = max(variants, key=lambda x: simple_score(x[1]))
        best_text = light_cleanup(best_text)

        x1, y1, x2, y2 = rect
        chosen_items.append({
            "x": x1,
            "y": y1,
            "text": best_text
        })

        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            debug_img,
            f"{i}:{best_variant}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        print(f"[{i}] {best_variant} | {best_text}")

    raw_lines = group_lines(chosen_items)
    raw_text = "\n".join(raw_lines)

    print("\n===== OCR RAW TEXT =====")
    print(raw_text)

    print("\nDang load correction model...")
    tokenizer, model = load_vietnamese_corrector()
    corrected_text = correct_text_with_lm(raw_text, tokenizer, model)
    corrected_text = light_cleanup(corrected_text)

    print("\n===== OCR CORRECTED TEXT =====")
    print(corrected_text)

    cv2.imwrite("debug_simple_pipeline.png", debug_img)
    print("\nDa luu: debug_simple_pipeline.png")


if __name__ == "__main__":
    main()