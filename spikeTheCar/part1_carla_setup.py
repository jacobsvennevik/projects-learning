#!/usr/bin/env python3
"""
Part 1: CARLA scenario and task setup for lane-keeping with obstacle avoidance.

This script:
- Connects to a running CARLA server (default: localhost:2000)
- Spawns an ego vehicle with forward RGB camera and optional LiDAR/Depth
- Spawns background traffic using Traffic Manager
- Runs multiple episodes across different weather/time conditions
- Logs basic metrics (collisions, lane invasions, distance) and saves sensor data

Prerequisites:
- Start CARLA server separately (e.g., ./CarlaUE4.sh -quality-level=Epic -opengl -world-port=2000)
- Ensure Python API is available (via installed carla module or EGG on PYTHONPATH)
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None  # Will only be used if available

try:
    import cv2
except Exception:
    cv2 = None  # Optional; image saving will fallback if not available


def try_append_carla_egg_to_sys_path() -> None:
    """Attempt to locate and append CARLA egg to sys.path if not installed as a package.

    Looks for EGG in common locations including CARLA_HOME/dist.
    """
    if "carla" in sys.modules:
        return

    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    candidates: List[str] = []
    carla_home = os.environ.get("CARLA_HOME")
    if carla_home:
        candidates.append(str(Path(carla_home) / "PythonAPI" / "carla" / "dist" / f"carla-*py{py_version}*.egg"))
        candidates.append(str(Path(carla_home) / "dist" / f"carla-*py{py_version}*.egg"))

    # Relative common locations (if running inside the CARLA repo tree)
    candidates.append(str(Path("..") / "carla" / "dist" / f"carla-*py{py_version}*.egg"))
    candidates.append(str(Path(".") / "carla" / "dist" / f"carla-*py{py_version}*.egg"))

    for pattern in candidates:
        try:
            matches = glob.glob(pattern)
        except Exception:
            matches = []
        if matches:
            sys.path.append(matches[0])
            break


try_append_carla_egg_to_sys_path()

try:
    import carla  # type: ignore
except Exception as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Failed to import carla. Ensure CARLA Python API is on PYTHONPATH or installed.\n"
        f"Original error: {exc}"
    )


# -------------------------- Configuration Models --------------------------- #


@dataclass
class CameraConfig:
    image_width: int = 800
    image_height: int = 600
    fov: float = 90.0
    sensor_tick: float = 0.05  # 20 FPS


@dataclass
class LidarConfig:
    enabled: bool = False
    channels: int = 32
    range: float = 50.0
    points_per_second: int = 100000
    rotation_frequency: float = 20.0
    upper_fov: float = 10.0
    lower_fov: float = -30.0
    sensor_tick: float = 0.05


@dataclass
class ScenarioConfig:
    name: str
    weather: carla.WeatherParameters
    episode_seconds: int = 60
    num_npc_vehicles: int = 30
    num_pedestrians: int = 0
    town_map: str = "Town03"  # Suburban-like


@dataclass
class RunConfig:
    output_root: Path
    host: str = "127.0.0.1"
    port: int = 2000
    tm_port: int = 8000
    sync: bool = True
    fixed_delta_seconds: float = 0.05
    seed: int = 42
    blueprint_filter: str = "vehicle.*"
    camera_config: CameraConfig = CameraConfig()
    lidar_config: LidarConfig = LidarConfig()


# ------------------------------ Sensor Manager ----------------------------- #


class SensorManager:
    def __init__(
        self,
        world: carla.World,
        output_dir: Path,
        camera_config: CameraConfig,
        lidar_config: LidarConfig,
    ) -> None:
        self.world = world
        self.blueprint_library = world.get_blueprint_library()
        self.output_dir = output_dir
        self.camera_config = camera_config
        self.lidar_config = lidar_config
        self.actors: List[carla.Actor] = []
        self.metrics: Dict[str, float] = {
            "collision_count": 0.0,
            "lane_invasion_count": 0.0,
        }

        (self.output_dir / "rgb").mkdir(parents=True, exist_ok=True)
        if lidar_config.enabled:
            (self.output_dir / "lidar").mkdir(parents=True, exist_ok=True)

    # ---------------------------- Sensor creation --------------------------- #

    def spawn_rgb_camera(self, parent: carla.Actor) -> carla.Sensor:
        bp = self.blueprint_library.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.camera_config.image_width))
        bp.set_attribute("image_size_y", str(self.camera_config.image_height))
        bp.set_attribute("fov", str(self.camera_config.fov))
        bp.set_attribute("sensor_tick", str(self.camera_config.sensor_tick))

        # Mount on windshield
        transform = carla.Transform(carla.Location(x=1.5, z=1.6))
        sensor: carla.Sensor = self.world.spawn_actor(bp, transform, attach_to=parent)
        self.actors.append(sensor)

        def _on_image(image: carla.Image) -> None:
            filename = self.output_dir / "rgb" / f"{image.frame:06d}.png"
            try:
                array = self._carla_image_to_bgr(image)
                if cv2 is not None and array is not None:
                    cv2.imwrite(str(filename), array)
                else:
                    image.save_to_disk(str(filename))
            except Exception:
                image.save_to_disk(str(filename))

        sensor.listen(_on_image)
        return sensor

    def spawn_lidar(self, parent: carla.Actor) -> Optional[carla.Sensor]:
        if not self.lidar_config.enabled:
            return None
        bp = self.blueprint_library.find("sensor.lidar.ray_cast")
        bp.set_attribute("channels", str(self.lidar_config.channels))
        bp.set_attribute("range", str(self.lidar_config.range))
        bp.set_attribute("points_per_second", str(self.lidar_config.points_per_second))
        bp.set_attribute("rotation_frequency", str(self.lidar_config.rotation_frequency))
        bp.set_attribute("upper_fov", str(self.lidar_config.upper_fov))
        bp.set_attribute("lower_fov", str(self.lidar_config.lower_fov))
        bp.set_attribute("sensor_tick", str(self.lidar_config.sensor_tick))

        transform = carla.Transform(carla.Location(x=0.0, z=2.1))
        sensor: carla.Sensor = self.world.spawn_actor(bp, transform, attach_to=parent)
        self.actors.append(sensor)

        def _on_lidar(point_cloud: carla.LidarMeasurement) -> None:
            filename = self.output_dir / "lidar" / f"{point_cloud.frame:06d}.ply"
            try:
                # Save as simple PLY (ASCII)
                with open(filename, "w", encoding="utf-8") as ply:
                    ply.write("ply\nformat ascii 1.0\n")
                    ply.write(f"element vertex {len(point_cloud)}\n")
                    ply.write("property float x\nproperty float y\nproperty float z\n")
                    ply.write("end_header\n")
                    for pt in point_cloud:
                        ply.write(f"{pt.point.x} {pt.point.y} {pt.point.z}\n")
            except Exception:
                pass

        sensor.listen(_on_lidar)
        return sensor

    def spawn_collision_sensor(self, parent: carla.Actor) -> carla.Sensor:
        bp = self.blueprint_library.find("sensor.other.collision")
        sensor: carla.Sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=parent)
        self.actors.append(sensor)

        def _on_collision(event: carla.CollisionEvent) -> None:
            self.metrics["collision_count"] += 1.0

        sensor.listen(_on_collision)
        return sensor

    def spawn_lane_invasion_sensor(self, parent: carla.Actor) -> carla.Sensor:
        bp = self.blueprint_library.find("sensor.other.lane_invasion")
        sensor: carla.Sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=parent)
        self.actors.append(sensor)

        def _on_lane_invasion(event: carla.LaneInvasionEvent) -> None:
            self.metrics["lane_invasion_count"] += 1.0

        sensor.listen(_on_lane_invasion)
        return sensor

    # ------------------------------ Utilities ------------------------------ #

    @staticmethod
    def _carla_image_to_bgr(image: carla.Image):
        if np is None:
            return None
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        bgr = array[:, :, :3][:, :, ::-1]
        return bgr

    def destroy(self) -> None:
        for actor in self.actors:
            try:
                actor.stop()  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                actor.destroy()
            except Exception:
                pass
        self.actors = []


# ---------------------------- Scenario Orchestrator ------------------------ #


class CarlaScenarioRunner:
    def __init__(self, run_config: RunConfig) -> None:
        self.cfg = run_config
        self.client = carla.Client(self.cfg.host, self.cfg.port)
        self.client.set_timeout(15.0)
        self.tm: Optional[carla.TrafficManager] = None

        # Runtime holders
        self.world: Optional[carla.World] = None
        self.ego_vehicle: Optional[carla.Vehicle] = None
        self.sensor_manager: Optional[SensorManager] = None
        self.npcs: List[carla.Actor] = []

    # ------------------------------- Lifecycle ----------------------------- #

    def setup_world(self, town: str) -> None:
        if self.client.get_world().get_map().name != town:
            self.world = self.client.load_world(town)
        else:
            self.world = self.client.get_world()

        assert self.world is not None

        settings = self.world.get_settings()
        settings.synchronous_mode = self.cfg.sync
        settings.fixed_delta_seconds = self.cfg.fixed_delta_seconds
        self.world.apply_settings(settings)

        self.tm = self.client.get_trafficmanager(self.cfg.tm_port)
        self.tm.set_global_distance_to_leading_vehicle(2.5)
        self.tm.global_percentage_speed_difference(10.0)
        if self.cfg.sync:
            self.tm.set_synchronous_mode(True)

    def spawn_ego(self) -> carla.Vehicle:
        assert self.world is not None
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter(self.cfg.blueprint_filter))
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "hero")

        spawn_points = self.world.get_map().get_spawn_points()
        random.seed(self.cfg.seed)
        spawn_point = random.choice(spawn_points)
        ego: carla.Vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.ego_vehicle = ego
        return ego

    def attach_sensors(self, scenario_output_dir: Path) -> None:
        assert self.world is not None and self.ego_vehicle is not None
        self.sensor_manager = SensorManager(
            self.world, scenario_output_dir, self.cfg.camera_config, self.cfg.lidar_config
        )
        self.sensor_manager.spawn_rgb_camera(self.ego_vehicle)
        self.sensor_manager.spawn_collision_sensor(self.ego_vehicle)
        self.sensor_manager.spawn_lane_invasion_sensor(self.ego_vehicle)
        self.sensor_manager.spawn_lidar(self.ego_vehicle)

    def spawn_npcs(self, num_vehicles: int, num_pedestrians: int) -> None:
        assert self.world is not None and self.tm is not None
        blueprint_library = self.world.get_blueprint_library()

        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        vehicles: List[carla.Vehicle] = []
        for idx, transform in enumerate(spawn_points[: num_vehicles + 5]):
            try:
                vehicle_bp = random.choice(blueprint_library.filter("vehicle.*"))
                vehicle: carla.Vehicle = self.world.spawn_actor(vehicle_bp, transform)
                vehicle.set_autopilot(True, self.tm.get_port())
                vehicles.append(vehicle)
            except Exception:
                continue

        self.npcs.extend(vehicles)

        # Pedestrians can be added with WalkerAI, omitted for simplicity here

    def set_weather(self, weather: carla.WeatherParameters) -> None:
        assert self.world is not None
        self.world.set_weather(weather)

    def enable_ego_autopilot(self, enabled: bool = True) -> None:
        assert self.ego_vehicle is not None and self.tm is not None
        self.ego_vehicle.set_autopilot(enabled, self.tm.get_port())

    # ------------------------------- Episodes ------------------------------- #

    def run_episode(self, scenario: ScenarioConfig, scenario_dir: Path) -> Dict[str, float]:
        assert self.world is not None
        # Apply weather
        self.set_weather(scenario.weather)

        # Spawn NPCs and Ego
        ego = self.spawn_ego()
        self.attach_sensors(scenario_dir)
        self.spawn_npcs(scenario.num_npc_vehicles, scenario.num_pedestrians)
        self.enable_ego_autopilot(True)

        # Step loop
        start_transform: carla.Transform = ego.get_transform()
        start_location: carla.Location = start_transform.location
        total_distance_m = 0.0
        sum_abs_lane_dev_m = 0.0
        max_abs_lane_dev_m = 0.0
        num_steps = 0

        start_time = time.time()
        end_time = start_time + scenario.episode_seconds
        try:
            while time.time() < end_time:
                if self.cfg.sync:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()

                # Approximate distance traveled since start
                ego_tf: carla.Transform = ego.get_transform()
                current_location: carla.Location = ego_tf.location
                total_distance_m = current_location.distance(start_location)

                # Lane deviation: lateral offset from nearest lane center
                try:
                    world_map: carla.Map = self.world.get_map()
                    wp: carla.Waypoint = world_map.get_waypoint(
                        current_location, project_to_road=True, lane_type=carla.LaneType.Driving
                    )
                    wp_tf: carla.Transform = wp.transform
                    # 2D vector from waypoint center to ego
                    dx = current_location.x - wp_tf.location.x
                    dy = current_location.y - wp_tf.location.y
                    # Right unit vector of lane direction (yaw)
                    yaw_rad = math.radians(wp_tf.rotation.yaw)
                    right_x = math.sin(yaw_rad)
                    right_y = -math.cos(yaw_rad)
                    lateral_offset = dx * right_x + dy * right_y
                    abs_off = abs(lateral_offset)
                    sum_abs_lane_dev_m += abs_off
                    if abs_off > max_abs_lane_dev_m:
                        max_abs_lane_dev_m = abs_off
                except Exception:
                    pass
                finally:
                    num_steps += 1

        finally:
            # Gather metrics
            sensor_metrics: Dict[str, float] = {}
            if self.sensor_manager is not None:
                sensor_metrics = dict(self.sensor_manager.metrics)

            avg_lane_dev_m = sum_abs_lane_dev_m / float(max(1, num_steps))

            metrics: Dict[str, float] = {
                "collisions": sensor_metrics.get("collision_count", 0.0),
                "lane_invasions": sensor_metrics.get("lane_invasion_count", 0.0),
                "distance_m": total_distance_m,
                "duration_s": float(scenario.episode_seconds),
                "avg_lane_dev_m": avg_lane_dev_m,
                "max_lane_dev_m": max_abs_lane_dev_m,
            }

            # Cleanup actors created for this episode
            self._cleanup_episode()

        return metrics

    # ------------------------------- Cleanup -------------------------------- #

    def _cleanup_episode(self) -> None:
        if self.sensor_manager is not None:
            self.sensor_manager.destroy()
            self.sensor_manager = None
        if self.ego_vehicle is not None:
            try:
                self.ego_vehicle.destroy()
            except Exception:
                pass
            self.ego_vehicle = None
        for actor in self.npcs:
            try:
                actor.destroy()
            except Exception:
                pass
        self.npcs = []

    def teardown(self) -> None:
        if self.world is not None and self.cfg.sync and self.tm is not None:
            try:
                self.tm.set_synchronous_mode(False)
            except Exception:
                pass
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
            except Exception:
                pass


# --------------------------------- Utility --------------------------------- #


def default_scenarios() -> List[ScenarioConfig]:
    return [
        ScenarioConfig(
            name="day_clear",
            weather=carla.WeatherParameters(
                cloudiness=0.0,
                precipitation=0.0,
                sun_altitude_angle=70.0,
                fog_density=0.0,
                wetness=0.0,
            ),
            episode_seconds=60,
            num_npc_vehicles=30,
        ),
        ScenarioConfig(
            name="night_clear",
            weather=carla.WeatherParameters(
                cloudiness=10.0,
                precipitation=0.0,
                sun_altitude_angle=-10.0,
                fog_density=0.0,
                wetness=0.0,
            ),
            episode_seconds=60,
            num_npc_vehicles=30,
        ),
        ScenarioConfig(
            name="day_rain",
            weather=carla.WeatherParameters(
                cloudiness=60.0,
                precipitation=60.0,
                precipitation_deposits=40.0,
                sun_altitude_angle=60.0,
                wetness=50.0,
                fog_density=10.0,
            ),
            episode_seconds=60,
            num_npc_vehicles=30,
        ),
        ScenarioConfig(
            name="night_heavy_rain",
            weather=carla.WeatherParameters(
                cloudiness=90.0,
                precipitation=90.0,
                precipitation_deposits=80.0,
                sun_altitude_angle=-15.0,
                wetness=90.0,
                fog_density=20.0,
            ),
            episode_seconds=60,
            num_npc_vehicles=30,
        ),
    ]


def ensure_output_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_metrics(metrics: List[Tuple[str, Dict[str, float]]], output_dir: Path) -> None:
    csv_path = output_dir / "metrics.csv"
    headers = [
        "scenario",
        "collisions",
        "lane_invasions",
        "distance_m",
        "duration_s",
        "avg_lane_dev_m",
        "max_lane_dev_m",
    ]
    lines = [",".join(headers)]
    for scenario_name, m in metrics:
        line = ",".join(
            [
                scenario_name,
                str(m.get("collisions", 0.0)),
                str(m.get("lane_invasions", 0.0)),
                f"{m.get('distance_m', 0.0):.2f}",
                f"{m.get('duration_s', 0.0):.2f}",
                f"{m.get('avg_lane_dev_m', 0.0):.3f}",
                f"{m.get('max_lane_dev_m', 0.0):.3f}",
            ]
        )
        lines.append(line)
    csv_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CARLA Part 1: Scenario and Task Setup")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port")
    parser.add_argument("--town", type=str, default="Town03", help="Town map to use")
    parser.add_argument("--duration", type=int, default=60, help="Seconds per episode")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output") / "part1_runs",
        help="Root folder for outputs",
    )
    parser.add_argument("--lidar", action="store_true", help="Enable LiDAR sensor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = ensure_output_dir(Path(args.output_root) / timestamp)

    run_cfg = RunConfig(
        output_root=base_output,
        host=args.host,
        port=args.port,
        tm_port=args.tm_port,
        seed=args.seed,
        lidar_config=LidarConfig(enabled=args.lidar),
    )

    runner = CarlaScenarioRunner(run_cfg)
    scenarios = default_scenarios()
    # Override town and duration from CLI
    scenarios = [
        ScenarioConfig(
            name=sc.name,
            weather=sc.weather,
            episode_seconds=args.duration,
            num_npc_vehicles=sc.num_npc_vehicles,
            num_pedestrians=sc.num_pedestrians,
            town_map=args.town,
        )
        for sc in scenarios
    ]

    all_metrics: List[Tuple[str, Dict[str, float]]] = []

    try:
        # Load town once; some weather may be applied per-episode
        runner.setup_world(town=args.town)

        for sc in scenarios:
            scenario_dir = ensure_output_dir(base_output / sc.name)
            print(f"Running scenario: {sc.name} in {sc.town_map}")
            episode_metrics = runner.run_episode(sc, scenario_dir)
            all_metrics.append((sc.name, episode_metrics))
            print(f"Finished {sc.name}: {episode_metrics}")

        save_metrics(all_metrics, base_output)
        print(f"Saved metrics to {base_output / 'metrics.csv'}")
    finally:
        runner.teardown()


if __name__ == "__main__":
    main()


