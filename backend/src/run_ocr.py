import os
import sys

# Add the current directory to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from src.ocr_engine import OCREngine

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_ocr.py <image_path>")
        return

    img_path = sys.argv[1]
    
    try:
        # Initialize engine and extract
        engine = OCREngine()
        text = engine.extract_text(img_path)
        
        # Tag the result so the main process can find it easily
        print("OCR_RESULT_START")
        print(text)
        print("OCR_RESULT_END")
    except Exception as e:
        print(f"Subprocess OCR Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
