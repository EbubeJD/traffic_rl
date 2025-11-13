# traffic_rl (CARLA TL Cameras + YOLO + ROI + Logging)

- Spawns one camera per traffic light head in a chosen group.
- Detects vehicles with YOLO.
- Counts only vehicles inside a manually drawn ROI polygon (per TL).
- Tracks vehicles to compute waiting times and stop-line crossings.
- Logs per-tick KPIs and per-crossing events to CSV, stable across runs.

## Quick start
1) Start CARLA server:
   ```
   CarlaUE4.exe -quality-level=Epic -carla-rpc-port=2000 -prefernvidiagpus
   ```

2) Install deps (Python 3.8+ recommended):
   ```
   pip install -r requirements.txt
   ```

3) Run the simulation:
   ```
   python run_sim.py
   ```

4) After first run, annotate per-lane ROI polygons (one per TL folder) using:
   ```
   python tools/roi_annotator.py <path_to_saved_frame.png> [<save_json_path>]
   ```

Outputs are in `outputs/<TOWN>/tl_<stable_id>/`.
