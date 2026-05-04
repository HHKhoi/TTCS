from src.preprocess import preprocess_image
from src.ocr_engine import OCREngine
from src.utils import save_text
from src.settings import SCALE, INPUT_PATH, OUTPUT_PATH


def run():
    img = preprocess_image(INPUT_PATH, SCALE)
    recognizer = OCREngine()
    text = recognizer.extract_text(img)
    print("OUTPUT:\n")
    print(text if text else "")
    save_text(text, OUTPUT_PATH)

if __name__ == "__main__":
    run()