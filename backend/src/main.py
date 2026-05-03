from src.preprocess import preprocess_image
from src.detector import TextDetector
from src.ocr_engine import OCREngine
from src.box_utils import sort_boxes, merge_boxes_into_lines, crop_box
from src.utils import save_text
from src.settings import SCALE, INPUT_PATH, OUTPUT_PATH
import cv2


def run():
    img = preprocess_image(INPUT_PATH, SCALE)
    detector = TextDetector()
    boxes = detector.detect_boxes(img)
    if not boxes:
        print("OUTPUT:\n")
        save_text("", OUTPUT_PATH)
        return
    boxes = sort_boxes(boxes)
    line_boxes = merge_boxes_into_lines(boxes, y_thresh=15)
    recognizer = OCREngine()
    lines = []
    for box in line_boxes:
        crop = crop_box(img, box, pad=10)
        if crop is None or crop.size == 0:
            continue
        text = recognizer.extract_text(crop)
        if text.strip():
            lines.append(text.strip())

    final_text = "\n".join(lines)
    print("OUTPUT:\n")
    print(final_text if final_text else "")
    save_text(final_text, OUTPUT_PATH)

if __name__ == "__main__":
    run()