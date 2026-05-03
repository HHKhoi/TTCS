def sort_boxes(boxes):
    return sorted(boxes, key=lambda b: (min(p[1] for p in b), min(p[0] for p in b)))

def merge_boxes_into_lines(boxes, y_thresh=15):
    if not boxes:
        return []
    lines = []
    current = []
    def y_center(box):
        return sum(p[1] for p in box) / 4.0
    for box in boxes:
        if not current:
            current.append(box)
            continue
        prev = current[-1]
        if abs(y_center(box) - y_center(prev)) <= y_thresh:
            current.append(box)
        else:
            lines.append(current)
            current = [box]
    if current:
        lines.append(current)
    merged = []
    for line in lines:
        xs = [p[0] for box in line for p in box]
        ys = [p[1] for box in line for p in box]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        merged.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return merged


def crop_box(img, box, pad=10):
    h, w = img.shape[:2]
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x1 = max(0, int(min(xs)) - pad)
    y1 = max(0, int(min(ys)) - pad)
    x2 = min(w, int(max(xs)) + pad)
    y2 = min(h, int(max(ys)) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]