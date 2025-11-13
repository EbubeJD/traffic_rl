import math

from config import IOU_MATCH_THR, TRACK_MAX_AGE

def iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua > 0 else 0.0

class TrackManager:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    @staticmethod
    def _centroid(box):
        x1,y1,x2,y2 = box
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    def update(self, dets, dt):
        unmatched = set(range(len(dets)))
        for tid, tr in list(self.tracks.items()):
            tr['age'] += 1
            tr['miss'] += 1
            best_j, best_iou = -1, -1.0
            for j in list(unmatched):
                i = iou(tr['box'], dets[j])
                if i > best_iou:
                    best_iou = i; best_j = j
            if best_iou >= IOU_MATCH_THR:
                j = best_j
                tr['box'] = dets[j]
                cx, cy = self._centroid(tr['box'])
                if tr['centroid'] is not None:
                    px, py = tr['centroid']
                    tr['speed_px'] = math.hypot(cx - px, cy - py) / max(dt,1e-6)
                else:
                    tr['speed_px'] = 0.0
                tr['centroid'] = (cx, cy)
                tr['miss'] = 0
                unmatched.remove(j)
        for j in unmatched:
            box = dets[j]
            cx, cy = self._centroid(box)
            self.tracks[self.next_id] = {
                'id': self.next_id,
                'box': box,
                'centroid': (cx, cy),
                'speed_px': 0.0,
                'age': 1,
                'miss': 0,
                'first_queue_time': None,
                'in_queue': False,
                'in_stripe_frames': 0,
                'crossed': False,
            }
            self.next_id += 1
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['miss'] > TRACK_MAX_AGE:
                del self.tracks[tid]
        return self.tracks
