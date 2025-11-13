import os, time
import numpy as np
import cv2
from queue import Queue, Empty

from config import (RES, SAVE_ROOT, SAVE_EVERY_N, DT,
                    STOP_SPEED_PX_THR, STRIPE_CONFIRM_FRAMES, TOWN)
from utils.carla_helpers import spawn_tl_camera, tl_stable_id, debug_draw_camera_ray
from utils.roi import load_polygon, point_in_poly, derive_stopline_from_roi
from log.tl_logger import TLLogger
from vision.detector import Detector
from vision.tracker import TrackManager

def make_sync_queue(sensor):
    q = Queue()
    sensor.listen(q.put)
    return q

class TLObserver:
    def __init__(self, world, tl_actor, bp_lib):
        self.world = world
        self.tl = tl_actor
        self.stable_id = tl_stable_id(world, self.tl)

        self.save_dir = os.path.join(SAVE_ROOT, TOWN, f"tl_{self.stable_id}")
        os.makedirs(self.save_dir, exist_ok=True)

        self.cam = spawn_tl_camera(world, self.tl, bp_lib)
        self.queue = make_sync_queue(self.cam)

        self.roi_path = os.path.join(self.save_dir, "roi.json")
        self.stopline_path = os.path.join(self.save_dir, "stopline.json")
        self.roi_poly = load_polygon(self.roi_path)
        self.stopline_poly = load_polygon(self.stopline_path)
        if self.roi_poly is None:
            print(f"[WARN] No ROI for TL {self.stable_id}. "
                  f"Annotate a saved frame with tools/roi_annotator.py")
        if self.stopline_poly is None and self.roi_poly is not None:
            self.stopline_poly = derive_stopline_from_roi(self.roi_poly, band_px=8)

        self.detector = Detector()
        self.tracker = TrackManager()
        self.logger = TLLogger(self.save_dir, self.tl, DT)

        debug_draw_camera_ray(world, self.cam)
        print(f"[cam] TL {self.stable_id} → {self.save_dir}")

    def destroy(self):
        for fn in (self.cam.stop, self.cam.destroy, self.logger.close):
            try: fn()
            except: pass

    @staticmethod
    def _centroid(box):
        x1,y1,x2,y2 = box
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    def get(self, frame_id):
        try:
            while True:
                ev = self.queue.get(timeout=1.0)
                if ev.frame == frame_id:
                    im = ev; break
        except Empty:
            return None

        h, w = im.height, im.width
        img = np.frombuffer(im.raw_data, dtype=np.uint8).reshape((h, w, 4))[:, :, :3].copy()

        if frame_id % SAVE_EVERY_N == 0:
            cv2.imwrite(os.path.join(self.save_dir, f"frame_{frame_id:06d}.png"), img)

        self.logger.update_state_timer(DT)

        dets = self.detector.detect_bgr(img)
        tracks = self.tracker.update(dets, DT)

        queue_count = 0
        waiting_times_now = []
        crossings = 0

        if self.roi_poly is not None:
            cv2.polylines(img, [self.roi_poly.reshape((-1,1,2))], True, (0,255,255), 2)
        if self.stopline_poly is not None:
            cv2.polylines(img, [self.stopline_poly.reshape((-1,1,2))], True, (255,255,0), 2)

        now_wall = time.time()

        for tid, tr in list(tracks.items()):
            x1,y1,x2,y2 = tr['box']
            cx, cy = map(int, self._centroid(tr['box']))
            speed_px = tr['speed_px']

            in_roi = (self.roi_poly is not None) and point_in_poly(cx, cy, self.roi_poly)
            is_stopped = speed_px < STOP_SPEED_PX_THR

            if in_roi and is_stopped:
                queue_count += 1
                if not tr['in_queue']:
                    tr['in_queue'] = True
                    if tr['first_queue_time'] is None:
                        tr['first_queue_time'] = now_wall
                        self.logger.log_arrival()
                waiting_times_now.append(max(0.0, now_wall - (tr['first_queue_time'] or now_wall)))
            else:
                if tr['in_queue'] and tr['first_queue_time'] is not None:
                    waiting_times_now.append(max(0.0, now_wall - tr['first_queue_time']))

            in_stripe = (self.stopline_poly is not None) and point_in_poly(cx, cy, self.stopline_poly)
            if in_stripe:
                tr['in_stripe_frames'] = min(999, tr['in_stripe_frames'] + 1)
            else:
                if tr['in_stripe_frames'] >= STRIPE_CONFIRM_FRAMES and tr['in_queue'] and not tr.get('crossed', False):
                    if tr['first_queue_time'] is not None:
                        wait_time = max(0.0, now_wall - tr['first_queue_time'])
                        self.logger.log_crossing(frame_id, tr['id'], wait_time)
                    tr['crossed'] = True
                    tr['in_queue'] = False
                    tr['first_queue_time'] = None
                    crossings += 1
                tr['in_stripe_frames'] = 0

            color = (0,255,0) if in_roi else (0,0,255)
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            cv2.circle(img, (cx,cy), 3, color, -1)

        self.logger.tick_log(frame_id, queue_count, waiting_times_now, crossings)

        if frame_id % SAVE_EVERY_N == 0:
            cv2.imwrite(os.path.join(self.save_dir, f"vis_{frame_id:06d}.png"), img)

        return {
            "frame": frame_id,
            "stable_id": self.stable_id,
            "state": self.logger.last_state,
            "time_in_state": self.logger.time_in_state,
            "queue": queue_count,
            "queue_ema": self.logger.queue_ema
        }
