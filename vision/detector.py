from ultralytics import YOLO
from config import YOLO_MODEL_PATH, VEHICLE_CLASSES

class Detector:
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.model = YOLO(model_path or YOLO_MODEL_PATH)
        self.device = device

    def detect_bgr(self, img_bgr):
        res = self.model.predict(img_bgr, verbose=False, device=self.device)[0]
        dets = []
        for b in res.boxes:
            cls = res.names[int(b.cls)]
            if cls not in VEHICLE_CLASSES:
                continue
            x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
            dets.append((x1,y1,x2,y2))
        return dets
