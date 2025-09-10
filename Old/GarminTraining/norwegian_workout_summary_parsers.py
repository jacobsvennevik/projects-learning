from dataclasses import dataclass
from typing import List, Optional
import re
from datetime import datetime, timedelta
import sqlite3
from norwegian_workout_parser import WorkoutLap, parse_workout_data

@dataclass
class WorkoutSummary:
    distance_km: float
    total_calories: int
    aerobic_te: float
    aerobic_te_desc: str
    anaerobic_te: float
    anaerobic_te_desc: str
    avg_hr: int
    max_hr: int
    total_time: str
    moving_time: str
    elapsed_time: str
    total_ascent: float
    total_descent: float
    min_elevation: float
    max_elevation: float
    avg_pace: str
    avg_moving_pace: str
    best_pace: str
    avg_cadence: int
    max_cadence: int
    avg_stride_length: float
    avg_temp: float
    min_temp: float
    max_temp: float

    @classmethod
    def from_text(cls, text: str) -> 'WorkoutSummary':
        def extract_value(pattern: str, text: str, default=None):
            match = re.search(pattern, text, re.MULTILINE)
            return match.group(1) if match else default

        def parse_float(text: str) -> float:
            try:
                return float(text.replace(',', '.').replace(' km', '').replace(' m', '').replace(' °C', ''))
            except (ValueError, AttributeError):
                return 0.0

        def parse_int(text: str) -> int:
            try:
                return int(text.replace(' bpm', '').replace(' spm', ''))
            except (ValueError, AttributeError):
                return 0

        def parse_te(text: str, type_str: str) -> tuple[float, str]:
            pattern = rf"{type_str}\n(\d+\.\d+)\s+([^\n]+)"
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                return float(match.group(1)), match.group(2)
            return 0.0, ""

        # Extract all values
        distance = parse_float(extract_value(r"Distanse\n([\d,.]+ km)", text, "0 km"))
        calories = parse_int(extract_value(r"Totale kalorier\n(\d+)", text, "0"))
        
        aerobic_te, aerobic_desc = parse_te(text, "Aerob")
        anaerobic_te, anaerobic_desc = parse_te(text, "Anaerob")
        
        avg_hr = parse_int(extract_value(r"Gjennomsnittlig puls\n(\d+ bpm)", text, "0"))
        max_hr = parse_int(extract_value(r"Makspuls\n(\d+ bpm)", text, "0"))
        
        total_time = extract_value(r"Tid\n([\d:]+)", text, "00:00")
        moving_time = extract_value(r"Tid i bevegelse\n([\d:]+)", text, "00:00")
        elapsed_time = extract_value(r"Medgått tid\n([\d:]+)", text, "00:00")
        
        total_ascent = parse_float(extract_value(r"Total stigning\n([\d,.]+ m)", text, "0"))
        total_descent = parse_float(extract_value(r"Totalt fall\n([\d,.]+ m)", text, "0"))
        min_elevation = parse_float(extract_value(r"Minimum høyde\n([\d,.]+ m)", text, "0"))
        max_elevation = parse_float(extract_value(r"Maksimum høyde\n([\d,.]+ m)", text, "0"))
        
        avg_pace = extract_value(r"Gjennomsnittlig tempo\n([\d:]+ /km)", text, "0:00 /km")
        avg_moving_pace = extract_value(r"Gjennomsnittlig bevegelsestempo\n([\d:]+ /km)", text, "0:00 /km")
        best_pace = extract_value(r"Beste tempo\n([\d:]+ /km)", text, "0:00 /km")
        
        avg_cadence = parse_int(extract_value(r"Gjennomsnittlig frekvens for løping\n(\d+ spm)", text, "0"))
        max_cadence = parse_int(extract_value(r"Maksimal frekvens for løping\n(\d+ spm)", text, "0"))
        avg_stride = parse_float(extract_value(r"Gjennomsnittlig skrittlengde\n([\d,.]+ m)", text, "0"))
        
        avg_temp = parse_float(extract_value(r"Gjennomsnittlig temperatur\n([\d,.]+ °C)", text, "0"))
        min_temp = parse_float(extract_value(r"Minimumstemperatur\n([\d,.]+ °C)", text, "0"))
        max_temp = parse_float(extract_value(r"Makstemperatur\n([\d,.]+ °C)", text, "0"))

        return cls(
            distance_km=distance,
            total_calories=calories,
            aerobic_te=aerobic_te,
            aerobic_te_desc=aerobic_desc,
            anaerobic_te=anaerobic_te,
            anaerobic_te_desc=anaerobic_desc,
            avg_hr=avg_hr,
            max_hr=max_hr,
            total_time=total_time,
            moving_time=moving_time,
            elapsed_time=elapsed_time,
            total_ascent=total_ascent,
            total_descent=total_descent,
            min_elevation=min_elevation,
            max_elevation=max_elevation,
            avg_pace=avg_pace,
            avg_moving_pace=avg_moving_pace,
            best_pace=best_pace,
            avg_cadence=avg_cadence,
            max_cadence=max_cadence,
            avg_stride_length=avg_stride,
            avg_temp=avg_temp,
            min_temp=min_temp,
            max_temp=max_temp
        )

@dataclass
class HeartRateZone:
    zone_number: int
    min_hr: int
    max_hr: int
    description: str
    duration: str
    percentage: float

@dataclass
class HeartRateZones:
    zones: List[HeartRateZone]

    @classmethod
    def from_text(cls, text: str) -> 'HeartRateZones':
        zones = []
        zone_pattern = r"Sone (\d+)\s+(\d+)(?:\s*-\s*(\d+))? bpm • ([^\n]+)\n\s*([\d:]+)\s+(\d+)%"
        
        matches = re.finditer(zone_pattern, text)
        for match in matches:
            zone_num = int(match.group(1))
            min_hr = int(match.group(2))
            max_hr = int(match.group(3)) if match.group(3) else 999
            desc = match.group(4)
            duration = match.group(5)
            percentage = float(match.group(6))
            
            zones.append(HeartRateZone(
                zone_number=zone_num,
                min_hr=min_hr,
                max_hr=max_hr,
                description=desc,
                duration=duration,
                percentage=percentage
            ))
        
        return cls(zones=sorted(zones, key=lambda x: x.zone_number, reverse=True))

def format_workout_summary(summary: WorkoutSummary) -> str:
    output = []
    output.append("🏃‍♂️ Workout Summary")
    output.append("=" * 50)
    
    # Basic stats
    output.append(f"Distance: {summary.distance_km:.2f} km")
    output.append(f"Duration: {summary.total_time}")
    output.append(f"Moving Time: {summary.moving_time}")
    output.append(f"Calories: {summary.total_calories}")
    
    # Training Effect
    output.append("\nTraining Effect:")
    output.append(f"Aerobic: {summary.aerobic_te} - {summary.aerobic_te_desc}")
    output.append(f"Anaerobic: {summary.anaerobic_te} - {summary.anaerobic_te_desc}")
    
    # Heart Rate
    output.append("\nHeart Rate:")
    output.append(f"Average: {summary.avg_hr} bpm")
    output.append(f"Maximum: {summary.max_hr} bpm")
    
    # Pace
    output.append("\nPace:")
    output.append(f"Average: {summary.avg_pace}")
    output.append(f"Moving Average: {summary.avg_moving_pace}")
    output.append(f"Best: {summary.best_pace}")
    
    # Elevation
    output.append("\nElevation:")
    output.append(f"Total Ascent: {summary.total_ascent:.1f}m")
    output.append(f"Total Descent: {summary.total_descent:.1f}m")
    output.append(f"Range: {summary.min_elevation:.1f}m - {summary.max_elevation:.1f}m")
    
    # Running Dynamics
    output.append("\nRunning Dynamics:")
    output.append(f"Average Cadence: {summary.avg_cadence} spm")
    output.append(f"Maximum Cadence: {summary.max_cadence} spm")
    output.append(f"Average Stride Length: {summary.avg_stride_length:.2f}m")
    
    # Temperature
    output.append("\nTemperature:")
    output.append(f"Average: {summary.avg_temp:.1f}°C")
    output.append(f"Range: {summary.min_temp:.1f}°C - {summary.max_temp:.1f}°C")
    
    return "\n".join(output)

def format_heart_rate_zones(zones: HeartRateZones) -> str:
    output = []
    output.append("❤️ Heart Rate Zones")
    output.append("=" * 50)
    
    for zone in zones.zones:
        hr_range = f"{zone.min_hr}-{zone.max_hr}" if zone.max_hr < 999 else f">{zone.min_hr}"
        output.append(f"\nZone {zone.zone_number} ({hr_range} bpm) - {zone.description}")
        output.append(f"Duration: {zone.duration}")
        output.append(f"Percentage: {zone.percentage}%")
    
    return "\n".join(output)

def init_database():
    conn = sqlite3.connect('workout_summaries.db')
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''CREATE TABLE IF NOT EXISTS workouts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT,
                  distance_km REAL,
                  total_calories INTEGER,
                  aerobic_te REAL,
                  aerobic_te_desc TEXT,
                  anaerobic_te REAL,
                  anaerobic_te_desc TEXT,
                  avg_hr INTEGER,
                  max_hr INTEGER,
                  total_time TEXT,
                  moving_time TEXT,
                  elapsed_time TEXT,
                  total_ascent REAL,
                  total_descent REAL,
                  min_elevation REAL,
                  max_elevation REAL,
                  avg_pace TEXT,
                  avg_moving_pace TEXT,
                  best_pace TEXT,
                  avg_cadence INTEGER,
                  max_cadence INTEGER,
                  avg_stride_length REAL,
                  avg_temp REAL,
                  min_temp REAL,
                  max_temp REAL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS heart_rate_zones
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  workout_id INTEGER,
                  zone_number INTEGER,
                  min_hr INTEGER,
                  max_hr INTEGER,
                  description TEXT,
                  duration TEXT,
                  percentage REAL,
                  FOREIGN KEY(workout_id) REFERENCES workouts(id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS laps
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  workout_id INTEGER,
                  lap_number INTEGER,
                  time TEXT,
                  total_time TEXT,
                  distance_km REAL,
                  avg_pace TEXT,
                  avg_gap TEXT,
                  avg_hr INTEGER,
                  max_hr INTEGER,
                  total_ascent INTEGER,
                  total_descent INTEGER,
                  avg_power TEXT,
                  avg_wkg TEXT,
                  max_power TEXT,
                  max_wkg TEXT,
                  avg_cadence INTEGER,
                  avg_ground_contact TEXT,
                  avg_gct_balance TEXT,
                  avg_stride_length REAL,
                  avg_vertical_oscillation TEXT,
                  avg_vertical_ratio TEXT,
                  calories INTEGER,
                  avg_temp REAL,
                  best_pace TEXT,
                  max_cadence INTEGER,
                  moving_time TEXT,
                  avg_moving_pace TEXT,
                  FOREIGN KEY(workout_id) REFERENCES workouts(id))''')
    
    conn.commit()
    return conn

def save_workout_to_db(conn, summary: WorkoutSummary, zones: HeartRateZones, laps: List[WorkoutLap]):
    c = conn.cursor()
    
    # Insert workout summary
    c.execute('''INSERT INTO workouts
                 (date, distance_km, total_calories, aerobic_te, aerobic_te_desc,
                  anaerobic_te, anaerobic_te_desc, avg_hr, max_hr, total_time,
                  moving_time, elapsed_time, total_ascent, total_descent,
                  min_elevation, max_elevation, avg_pace, avg_moving_pace,
                  best_pace, avg_cadence, max_cadence, avg_stride_length,
                  avg_temp, min_temp, max_temp)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 summary.distance_km, summary.total_calories,
                 summary.aerobic_te, summary.aerobic_te_desc,
                 summary.anaerobic_te, summary.anaerobic_te_desc,
                 summary.avg_hr, summary.max_hr, summary.total_time,
                 summary.moving_time, summary.elapsed_time,
                 summary.total_ascent, summary.total_descent,
                 summary.min_elevation, summary.max_elevation,
                 summary.avg_pace, summary.avg_moving_pace,
                 summary.best_pace, summary.avg_cadence,
                 summary.max_cadence, summary.avg_stride_length,
                 summary.avg_temp, summary.min_temp, summary.max_temp))
    
    workout_id = c.lastrowid
    
    # Insert heart rate zones
    for zone in zones.zones:
        c.execute('''INSERT INTO heart_rate_zones
                     (workout_id, zone_number, min_hr, max_hr, description,
                      duration, percentage)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (workout_id, zone.zone_number, zone.min_hr,
                     zone.max_hr, zone.description,
                     zone.duration, zone.percentage))
    
    # Insert laps
    for lap in laps:
        c.execute('''INSERT INTO laps
                     (workout_id, lap_number, time, total_time, distance_km,
                      avg_pace, avg_gap, avg_hr, max_hr, total_ascent,
                      total_descent, avg_power, avg_wkg, max_power, max_wkg,
                      avg_cadence, avg_ground_contact, avg_gct_balance,
                      avg_stride_length, avg_vertical_oscillation,
                      avg_vertical_ratio, calories, avg_temp, best_pace,
                      max_cadence, moving_time, avg_moving_pace)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (workout_id, lap.lap_number, lap.time, lap.total_time,
                     lap.distance_km, lap.avg_pace, lap.avg_gap, lap.avg_hr,
                     lap.max_hr, lap.total_ascent, lap.total_descent,
                     lap.avg_power, lap.avg_wkg, lap.max_power, lap.max_wkg,
                     lap.avg_run_cadence, lap.avg_ground_contact,
                     lap.avg_gct_balance, lap.avg_stride_length,
                     lap.avg_vertical_oscillation, lap.avg_vertical_ratio,
                     lap.total_calories, lap.avg_temperature, lap.best_pace,
                     lap.max_run_cadence, lap.moving_time, lap.avg_moving_pace))
    
    conn.commit()

def get_input(prompt: str) -> str:
    print(prompt)
    data = []
    while True:
        line = input()
        if line.strip() == "" and data:
            break
        data.append(line)
    return "\n".join(data)

def main():
    print("Welcome to the Norwegian Workout Data Parser!")
    print("=" * 50)
    
    # Initialize database
    conn = init_database()
    
    # Get lap data
    print("\nStep 1: Enter the lap data")
    print("(press Enter twice when finished)")
    laps_text = get_input("")
    
    # Get workout summary data
    print("\nStep 2: Enter the workout summary data")
    print("(press Enter twice when finished)")
    workout_text = get_input("")
    
    # Get heart rate zones data
    print("\nStep 3: Enter the heart rate zones data")
    print("(press Enter twice when finished)")
    zones_text = get_input("")
    
    # Parse the data
    try:
        laps = parse_workout_data(laps_text)[0]  # Get laps from the tuple
        summary = WorkoutSummary.from_text(workout_text)
        zones = HeartRateZones.from_text(zones_text)
        
        # Save to database
        save_workout_to_db(conn, summary, zones, laps)
        
        # Display the parsed data
        print("\n" + format_workout_summary(summary))
        print("\n" + format_heart_rate_zones(zones))
        print("\nLap Data:")
        for lap in laps:
            print(f"\nLap {lap.lap_number}:")
            print(f"Time: {lap.time} ({lap.total_time})")
            print(f"Distance: {lap.distance_km:.2f}km")
            print(f"Pace: {lap.avg_pace} (Moving: {lap.avg_moving_pace})")
            print(f"HR: {lap.avg_hr} (Max: {lap.max_hr})")
            if lap.avg_run_cadence:
                print(f"Cadence: {lap.avg_run_cadence} spm")
        
        print("\nWorkout data has been saved to the database!")
        
    except Exception as e:
        print(f"\nError parsing workout data: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main() 