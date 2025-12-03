import os, csv
import numpy as np

from config import EMA_ALPHA_QUEUE, EMA_ALPHA_ARRIVAL, EMA_ALPHA_DISCH

class TLLogger:
    def __init__(self, save_dir, tl_actor, dt):
        self.save_dir = save_dir
        self.tl_actor = tl_actor
        self.dt = dt
        os.makedirs(self.save_dir, exist_ok=True)
        self.tick_csv = open(os.path.join(save_dir, "ticks.csv"), "w", newline="")
        self.cross_csv = open(os.path.join(save_dir, "crossings.csv"), "w", newline="")
        self.tick_w = csv.writer(self.tick_csv)
        self.cross_w = csv.writer(self.cross_csv)
        self.tick_w.writerow([
            "t_sec","frame","state","time_in_state",
            "queue_count","queue_ema",
            "avg_wait","max_wait","num_long_wait_60s",
            "arrival_ema","discharge_ema"
        ])
        self.cross_w.writerow(["t_sec","frame","track_id","wait_time_sec"])

        self.queue_ema = 0.0
        self.arrival_ema = 0.0
        self.discharge_ema = 0.0
        self.last_state = None
        self.time_in_state = 0.0
        self.total_time = 0.0
        self.recent_arrivals = 0
        self.recent_discharges = 0
        self._sec_accum = 0.0

        # Expose metrics for RL environment
        self.avg_wait = 0.0
        self.max_wait = 0.0
        self.num_long_wait_60s = 0

    def get_state_name(self):
        try:
            return self.tl_actor.get_state().name
        except:
            return str(self.tl_actor.get_state())

    def update_state_timer(self, dt):
        st = self.get_state_name()
        if st == self.last_state:
            self.time_in_state += dt
        else:
            self.last_state = st
            self.time_in_state = dt

    def tick_log(self, frame, queue_count, waiting_now, crossings_this_tick):
        self.total_time += self.dt
        self._sec_accum += self.dt

        self.queue_ema = EMA_ALPHA_QUEUE*queue_count + (1-EMA_ALPHA_QUEUE)*self.queue_ema
        self.recent_discharges += crossings_this_tick

        avg_wait = float(np.mean(waiting_now)) if waiting_now else 0.0
        max_wait = float(np.max(waiting_now)) if waiting_now else 0.0
        long_waiters = sum(1 for w in waiting_now if w >= 60.0)

        # Expose metrics for RL environment
        self.avg_wait = avg_wait
        self.max_wait = max_wait
        self.num_long_wait_60s = long_waiters

        if self._sec_accum >= 1.0 - 1e-6:
            self.discharge_ema = EMA_ALPHA_DISCH*(self.recent_discharges / self._sec_accum) + (1-EMA_ALPHA_DISCH)*self.discharge_ema
            self.arrival_ema = EMA_ALPHA_ARRIVAL*(self.recent_arrivals / self._sec_accum) + (1-EMA_ALPHA_ARRIVAL)*self.arrival_ema
            self.recent_arrivals = 0
            self.recent_discharges = 0
            self._sec_accum = 0.0

        self.tick_w.writerow([
            round(self.total_time,3), frame, self.last_state, round(self.time_in_state,3),
            int(queue_count), round(self.queue_ema,3),
            round(avg_wait,3), round(max_wait,3), int(long_waiters),
            round(self.arrival_ema,3), round(self.discharge_ema,3)
        ])

    def log_crossing(self, frame, track_id, wait_time):
        self.cross_w.writerow([round(self.total_time,3), frame, track_id, round(wait_time,3)])
        self.recent_discharges += 1

    def log_arrival(self):
        self.recent_arrivals += 1

    def close(self):
        try: self.tick_csv.close()
        except: pass
        try: self.cross_csv.close()
        except: pass
