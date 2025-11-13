import json
import numpy as np
import cv2

def load_polygon(json_path):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        poly = data.get("polygon", [])
        return np.array(poly, dtype=np.int32) if len(poly) >= 3 else None
    except Exception:
        return None

def point_in_poly(cx, cy, poly_np):
    poly = poly_np.reshape((-1,1,2)) if poly_np.ndim == 2 else poly_np
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0

def derive_stopline_from_roi(roi_np, band_px=8):
    pts = roi_np.reshape(-1, 2)
    pairs = [(pts[i], pts[(i+1) % len(pts)]) for i in range(len(pts))]
    edge = max(pairs, key=lambda ab: (ab[0][1] + ab[1][1]) / 2.0)
    (x1,y1),(x2,y2) = edge
    v = np.array([x2-x1, y2-y1], dtype=np.float32)
    n = np.array([-v[1], v[0]], dtype=np.float32)
    n = n / (np.linalg.norm(n) + 1e-6)
    if n[1] > 0:
        n = -n
    p1 = np.array([x1,y1], np.float32)
    p2 = np.array([x2,y2], np.float32)
    p3 = p2 + n*band_px
    p4 = p1 + n*band_px
    quad = np.stack([p1,p2,p3,p4], axis=0).astype(np.int32)
    return quad
