import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
from src.preprocess import preprocess_image
from src.detector import TextDetector
from src.box_utils import sort_boxes, merge_boxes_into_lines, crop_box
from src.ocr_engine import OCREngine
import re
from jiwer import wer
from tqdm import tqdm


print("START EVALUATE")

def normalize(text):
    text = text.lower()
    text = re.sub(r'[.,]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def levenshtein(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[-1][-1]


def run_full_ocr(image_path, detector, recognizer):
    if not os.path.exists(image_path):
        return ""
    img = preprocess_image(image_path, scale=2)
    boxes = detector.detect_boxes(img)
    if not boxes:
        return ""
    boxes = sort_boxes(boxes)
    line_boxes = merge_boxes_into_lines(boxes, y_thresh=10)
    lines = []
    for box in line_boxes:
        crop = crop_box(img, box, pad=18)
        if crop is None or crop.size == 0:
            continue
        text = recognizer.extract_text(crop)
        if text.strip():
            lines.append(text.strip())
    return " ".join(lines).strip()

def evaluate(test_file, max_samples=None):
    detector = TextDetector()
    recognizer = OCREngine()
    total_cer = 0
    total_wer = 0
    exact_match = 0
    near_match = 0
    count = 0
    with open(test_file, encoding="utf-8") as f:
        lines = f.readlines()
    if max_samples:
        lines = lines[:max_samples]
    for idx, line in enumerate(tqdm(lines, desc="Processing")):
        try:
            path, gt = line.strip().split("|", 1)
            path = path.replace("\\", "/")
            pred = run_full_ocr(path, detector, recognizer)
            if pred.strip() == "":
                continue
            gt_n = normalize(gt)
            pred_n = normalize(pred)
            dist = levenshtein(gt_n, pred_n)
            cer = dist / max(1, len(gt_n))
            w = wer(gt_n, pred_n)
            similarity = 1 - cer
            if pred_n == gt_n:
                exact_match += 1
            if similarity > 0.97:
                near_match += 1
            total_cer += cer
            total_wer += w
            count += 1
            if idx < 3:
                print("\n--- SAMPLE ---")
                print("GT  :", gt_n[:120])
                print("PRED:", pred_n[:120])
                print("SIM :", round(similarity, 4))
        except Exception as e:
            print(f"[ERROR] {path} -> {e}")
            continue
    avg_cer = total_cer / count if count else 0
    avg_wer = total_wer / count if count else 0
    exact_acc = exact_match / count * 100 if count else 0
    near_acc = near_match / count * 100 if count else 0
    print("\n========== EVALUATION SUMMARY ==========\n")
    print(f"{'Metric':<25} {'Value'}")
    print("-" * 45)
    print(f"{'Total samples':<25} {count}")
    print(f"{'Exact match (%)':<25} {exact_acc:.2f}")
    print(f"{'Near match (%)':<25} {near_acc:.2f}")
    print(f"{'Average CER':<25} {avg_cer:.4f}")
    print(f"{'Average WER':<25} {avg_wer:.4f}")
    print("\n========================================\n")


if __name__ == "__main__":
    print("RUNNING MAIN")
    evaluate("data.txt")