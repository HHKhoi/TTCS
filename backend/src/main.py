from src.preprocess import preprocess_image
from src.detector import TextDetector
from src.ocr_engine import OCREngine
from src.box_utils import sort_boxes, crop_box
from src.utils import save_text
from src.settings import INPUT_PATH, OUTPUT_PATH

def run():
    recognizer = OCREngine()
    detector = TextDetector()
    
    # phóng to ảnh lên 2 lần
    img = preprocess_image(INPUT_PATH, 2)
    
    # Tìm các khung chữ trong ảnh
    boxes = detector.detect_boxes(img)
    
    if not boxes:
        print("OUTPUT")
        save_text("", OUTPUT_PATH)
        return
        
    # Sắp xếp các hộp chữ theo thứ tự từ trên xuống dưới
    boxes = sort_boxes(boxes)
    
    lines = []
    
    for box in boxes:
        # Cắt khung chữ
        crop = crop_box(img, box, pad=10)
        if crop is None or crop.size == 0:
            continue

        # Nhận diện văn bản từ khung đã cắt
        text = recognizer.extract_text(crop)
        
        # Nếu đọc được chữ thêm vào danh sách kết quả
        if text.strip():
            lines.append(text.strip())

    # Gộp các dòng chữ bằng dấu xuống dòng
    final_text = "\n".join(lines)
    print("OUTPUT")
    if final_text:
        print(final_text)
    else:
        print("Không đọc được chữ nào")
    save_text(final_text, OUTPUT_PATH)

if __name__ == "__main__":
    run()