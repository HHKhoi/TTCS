import re
import cv2
import unicodedata
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

try:
    import Levenshtein
    dist = Levenshtein.distance
except ImportError:
    def dist(a, b):
        la, lb = len(a), len(b)
        dp = [[0] * (lb + 1) for _ in range(la + 1)]
        for i in range(la + 1):
            dp[i][0] = i
        for j in range(lb + 1):
            dp[0][j] = j
        for i in range(1, la + 1):
            for j in range(1, lb + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost
                )
        return dp[la][lb]


IMAGE_PATH = r"D:\OCR-Project\data/images/Screenshot 2026-03-10 212957.png"
VI_DICT_PATH = r"D:\OCR-Project\data\vietnamese_words.txt"
EN_DICT_PATH = r"D:\OCR-Project\data\english_words.txt"
PROTECTED_PATH = r"D:\OCR-Project\data\protected_tokens.txt"


def load_words(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {x.strip().lower() for x in f if x.strip()}
    except FileNotFoundError:
        return set()


VI_WORDS = load_words(VI_DICT_PATH)
EN_WORDS = load_words(EN_DICT_PATH)
PROTECTED = load_words(PROTECTED_PATH)
ALL_WORDS = VI_WORDS | EN_WORDS


def norm(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = (
        text.replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'")
            .replace("–", "-").replace("—", "-").replace("−", "-")
            .replace("\xa0", " ")
    )
    return re.sub(r"[ \t]+", " ", text).strip()


import re
import unicodedata

def norm(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = (
        text.replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'")
            .replace("–", "-").replace("—", "-").replace("−", "-")
            .replace("\xa0", " ")
    )
    return re.sub(r"[ \t]+", " ", text).strip()


import re
import unicodedata


def norm(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = (
        text.replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'")
            .replace("–", "-").replace("—", "-").replace("−", "-")
            .replace("\xa0", " ")
    )
    return re.sub(r"[ \t]+", " ", text).strip()


def cleanup(text):
    if not text:
        return ""

    text = norm(text)

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    text = " ".join(lines)

    prev = None
    while prev != text:
        prev = text
        text = re.sub(r'"([^"\n]+?)\s+"([^"\n]+?)"', r'"\1 \2"', text)

    # là"Cúp -> là "Cúp
    text = re.sub(r'(?<=[A-Za-zÀ-ỹ0-9])"', ' "', text)

    # " Cúp -> "Cúp
    text = re.sub(r'"\s+([A-Za-zÀ-ỹ0-9])', r'"\1', text)

    # "Cúp Anh " -> "Cúp Anh"
    # chỉ xóa space ở bên trong quote trước dấu " đóng
    text = re.sub(r'"([^"\n]*?)\s+"', r'"\1"', text)

    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])([^\s,.;:!?])", r"\1 \2", text)

    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "-", text)

    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r'"\s+\)', '")', text)

    text = re.sub(r'"{2,}', '"', text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"\s{2,}", " ", text)

    if text.count('"') % 2 != 0:
        i = text.rfind('"')
        if i != -1:
            text = text[:i] + text[i + 1:]

    return text.strip()


def score_text(text):
    if not text:
        return -1e9
    text = norm(text)
    words = re.findall(r"[A-Za-zÀ-ỹ]+", text)
    valid = sum(
        w.lower() in VI_WORDS or w.lower() in EN_WORDS or w.lower() in PROTECTED
        for w in words
    )
    weird = len(re.findall(r"[^0-9A-Za-zÀ-ỹ\s,.;:!?()\"'/%+\-_=:/]", text))
    return len(text) * 0.05 + valid * 0.8 - weird * 3 - text.count("�") * 5 - text.count("?") * 2


def load_recognizer(device="cpu"):
    cfg = Cfg.load_config_from_name("vgg_transformer")
    cfg["device"] = device
    cfg["predictor"]["beamsearch"] = False
    return Predictor(cfg)


def order_boxes(boxes):
    return sorted(boxes, key=lambda b: (min(p[1] for p in b), min(p[0] for p in b)))


def crop_box(img, box):
    xs = [int(p[0]) for p in box]
    ys = [int(p[1]) for p in box]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w, h = x2 - x1, y2 - y1
    px, py = max(6, int(w * 0.03)), max(4, int(h * 0.08))
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(img.shape[1], x2 + px), min(img.shape[0], y2 + py)
    return img[y1:y2, x1:x2], (x1, y1)


def valid_crop(crop):
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    return h >= 16 and w >= 25 and h <= 220


def make_variants(crop, rec):
    h, w = crop.shape[:2]
    scale = 3.5 if w < 900 else 2.5

    base = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    sharp = cv2.filter2D(
        gray, -1,
        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    )
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    contrast = cv2.convertScaleAbs(gray, alpha=1.8, beta=20)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    imgs = [gray, sharp, thresh, contrast, otsu]
    return [norm(rec.predict(Image.fromarray(x))) for x in imgs]


def should_skip(word):
    return (
        not word
        or word.isdigit()
        or word.lower() in PROTECTED
        or (word.isupper() and len(word) <= 6)
        or len(word) <= 2
        or bool(re.search(r"\d", word))
        or any(ch in word for ch in "_-/=+")
    )


def same_shape(a, b):
    return len(a) >= 3 and len(b) >= 3 and a[0] == b[0] and abs(len(a) - len(b)) <= 2


def bonus(w):
    w = w.lower()
    if w in PROTECTED:
        return 0.35
    if w in VI_WORDS:
        return 0.20
    if w in EN_WORDS:
        return 0.18
    return 0.0


def candidates(word, topk=6, max_dist=2):
    out = []
    for v in ALL_WORDS:
        if abs(len(v) - len(word)) > 2:
            continue
        d = dist(word, v)
        if d > max_dist:
            continue
        s = -d
        if same_shape(word, v):
            s += 0.8
        if word and v and word[0] == v[0]:
            s += 0.4
        s += bonus(v)
        out.append((v, s))

    out.sort(key=lambda x: x[1], reverse=True)
    res = [w for w, _ in out[:topk]]
    return [word] + [x for x in res if x != word]


def correct_token(tok):
    m = re.match(r"^([^A-Za-zÀ-ỹ0-9]*)([A-Za-zÀ-ỹ0-9]+)([^A-Za-zÀ-ỹ0-9]*)$", tok)
    if not m:
        return tok

    pre, core, suf = m.groups()
    low = core.lower()

    if should_skip(core) or low in VI_WORDS or low in EN_WORDS or low in PROTECTED:
        return tok

    if bool(re.search(r"[A-Za-z]", core)) and bool(re.search(r"[À-ỹ]", core)):
        return tok

    cands = candidates(low)
    best = cands[0] if cands else low

    if best == low or dist(low, best) > 2:
        return tok

    if core.isupper():
        best = best.upper()
    elif core[:1].isupper():
        best = best.capitalize()

    return f"{pre}{best}{suf}"


def spell_correct(text):
    text = " ".join(correct_token(x) for x in norm(text).split())
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def group_lines(items, y_threshold=22):
    if not items:
        return []

    items = sorted(items, key=lambda x: (x["y"], x["x"]))
    groups, cur = [], [items[0]]

    for item in items[1:]:
        if abs(item["y"] - cur[-1]["y"]) <= y_threshold:
            cur.append(item)
        else:
            groups.append(cur)
            cur = [item]

    groups.append(cur)

    return [
        norm(" ".join(x["text"] for x in sorted(g, key=lambda z: z["x"]) if x["text"].strip()))
        for g in groups
    ]


def main():
    print("VI_WORDS:", len(VI_WORDS))
    print("EN_WORDS:", len(EN_WORDS))
    print("PROTECTED:", len(PROTECTED))

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {IMAGE_PATH}")

    detector = PaddleOCR(lang="vi")
    recognizer = load_recognizer("cpu")

    det = detector.predict(img)
    if not det:
        print("Khong co ket qua detect.")
        return

    boxes = det[0].get("dt_polys", [])
    if not boxes:
        print("Khong tim thay text box.")
        return

    items = []
    for box in order_boxes(boxes):
        crop, (x, y) = crop_box(img, box)
        if not valid_crop(crop):
            continue

        text = max(make_variants(crop, recognizer), key=score_text)
        text = spell_correct(text)

        if text:
            items.append({"x": x, "y": y, "text": text})

    result = cleanup("\n".join(group_lines(items)))

    print("\n===== OCR OUTPUT =====\n")
    print(result)


if __name__ == "__main__":
    main()