import os

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OCR_LANG = "en"
DET_LANG = "en"
USE_GPU = False

SCALE = 2
INPUT_PATH = os.path.join(_BACKEND_DIR, "dataset", "697.png")
OUTPUT_PATH = os.path.join(_BACKEND_DIR, "output", "output.txt")