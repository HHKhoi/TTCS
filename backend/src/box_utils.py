# Sắp xếp các boxes theo thứ tự từ trái sang phải từ trên xuống dưới
def sort_boxes(boxes):
    def get_top_left(box):
        x = box[0][0]
        y = box[0][1]
        return (y, x)
    return sorted(boxes, key=get_top_left)

def crop_box(img, box, pad=10):
    # Lấy chiều cao và độ dài của ảnh
    h = img.shape[0]
    w = img.shape[1]

    # Danh sách toạ độ x,y của 4 điểm của khung chữ
    xs = []
    ys = []
    for p in box:
        xs.append(p[0])
        ys.append(p[1])
        
    # Tính biên mới sau khi cộng thêm pad
    x1 = int(min(xs)) - pad
    y1 = int(min(ys)) - pad
    x2 = int(max(xs)) + pad
    y2 = int(max(ys)) + pad
    
    # Kiểm tra lại toạ độ mới có vượt ra ngoài viền ảnh gốc hay không
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    if x2 > w: x2 = w
    if y2 > h: y2 = h
    
    # Cắt khung chữ
    return img[y1:y2, x1:x2]