import json, os, sys
import numpy as np
import cv2

def annotate_roi(image_path, save_json_path):
    img0 = cv2.imread(image_path)
    if img0 is None:
        raise RuntimeError(f"Cannot open {image_path}")
    img = img0.copy()
    pts = []

    def redraw():
        nonlocal img
        img = img0.copy()
        for i, (x,y) in enumerate(pts):
            cv2.circle(img, (x,y), 3, (0,255,0), -1)
            if i > 0:
                cv2.line(img, tuple(pts[i-1]), (x,y), (0,255,0), 2)
        if len(pts) >= 3:
            cv2.polylines(img, [np.array(pts, np.int32).reshape((-1,1,2))], True, (0,255,255), 2)
        cv2.imshow("ROI", img)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([x, y])
            redraw()

    cv2.imshow("ROI", img)
    cv2.setMouseCallback("ROI", on_click)
    print("Left-click to add points. Keys: 's' save | 'r' reset | 'q' quit.")

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == ord('r'):
            pts = []
            redraw()
        elif k == ord('s'):
            if len(pts) >= 3:
                os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
                with open(save_json_path, "w") as f:
                    json.dump({"polygon": pts}, f)
                print(f"Saved ROI with {len(pts)} points → {save_json_path}")
                break
            else:
                print("Need at least 3 points.")
        elif k == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) not in (2,3):
        print("Usage: python tools/roi_annotator.py <image_path> [<save_json_path>]")
        sys.exit(1)
    img_path = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) == 3 else os.path.join(os.path.dirname(img_path), "roi.json")
    annotate_roi(img_path, save_path)
