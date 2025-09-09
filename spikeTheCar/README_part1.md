### Part 1: CARLA Scenario Setup

This repo contains a script to set up and run CARLA driving scenarios for lane keeping with obstacle avoidance, capturing sensor data and basic metrics.

### Prerequisites
- CARLA simulator running (e.g., `./CarlaUE4.sh -quality-level=Epic -opengl -world-port=2000`)
- Python 3.9+ and the CARLA Python API available on `PYTHONPATH` (via EGG or pip)

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# If needed (and matches your CARLA build):
# pip install carla==0.9.15
```

If using the EGG provided by CARLA, set `CARLA_HOME` env var or ensure the EGG path is discoverable. The script tries to auto-locate the EGG in common locations.

### Run
```bash
python part1_carla_setup.py --town Town03 --duration 60 --lidar
```

Key flags:
- `--host`, `--port`: CARLA server address (default `127.0.0.1:2000`)
- `--tm-port`: Traffic Manager port (default `8000`)
- `--town`: CARLA map (default `Town03`)
- `--duration`: seconds per episode (default `60`)
- `--lidar`: enable LiDAR capture (disabled by default)

Outputs (timestamped directory under `output/part1_runs/`):
- `rgb/`: PNGs from the forward camera
- `lidar/`: PLY point clouds (if LiDAR is enabled)
- `metrics.csv`: per-scenario collisions, lane invasions, distance, duration, average/max lane deviation (m)

### Notes
- The ego vehicle uses CARLA autopilot in Part 1 to focus on data and evaluation harness. Controllers (ANN/SNN) come in later parts.
- Scenarios include day/night and rain variants to probe robustness.

