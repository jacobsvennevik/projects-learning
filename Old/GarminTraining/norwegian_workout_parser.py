import dataclasses
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
import re

@dataclasses.dataclass
class WorkoutLap:
    lap_number: int
    time: str
    total_time: str
    distance_km: float
    avg_pace: str
    avg_gap: str
    avg_hr: Optional[int]
    max_hr: Optional[int]
    total_ascent: int
    total_descent: int
    avg_power: str
    avg_wkg: str
    max_power: str
    max_wkg: str
    avg_run_cadence: Optional[int]
    avg_ground_contact: str
    avg_gct_balance: str
    avg_stride_length: float
    avg_vertical_oscillation: str
    avg_vertical_ratio: str
    total_calories: int
    avg_temperature: float
    best_pace: str
    max_run_cadence: Optional[int]
    moving_time: str
    avg_moving_pace: str
    avg_stride_speed_loss: str
    avg_stride_speed_loss_percent: str

    @classmethod
    def from_row(cls, row_data: List[str]) -> 'WorkoutLap':
        def parse_time(time_str: str) -> str:
            return time_str.strip()

        def parse_float(val: str) -> float:
            try:
                return float(val.replace(',', '.'))
            except ValueError:
                return 0.0

        def parse_int(val: str) -> Optional[int]:
            try:
                return int(val)
            except ValueError:
                return None

        def clean_str(val: str) -> str:
            return '--' if val.strip() == '--' else val.strip()

        return cls(
            lap_number=int(row_data[0]),
            time=parse_time(row_data[1]),
            total_time=parse_time(row_data[2]),
            distance_km=parse_float(row_data[3]),
            avg_pace=clean_str(row_data[4]),
            avg_gap=clean_str(row_data[5]),
            avg_hr=parse_int(row_data[6]),
            max_hr=parse_int(row_data[7]),
            total_ascent=parse_int(row_data[8]) or 0,
            total_descent=parse_int(row_data[9]) or 0,
            avg_power=clean_str(row_data[10]),
            avg_wkg=clean_str(row_data[11]),
            max_power=clean_str(row_data[12]),
            max_wkg=clean_str(row_data[13]),
            avg_run_cadence=parse_int(row_data[14]),
            avg_ground_contact=clean_str(row_data[15]),
            avg_gct_balance=clean_str(row_data[16]),
            avg_stride_length=parse_float(row_data[17]),
            avg_vertical_oscillation=clean_str(row_data[18]),
            avg_vertical_ratio=clean_str(row_data[19]),
            total_calories=parse_int(row_data[20]) or 0,
            avg_temperature=parse_float(row_data[21]),
            best_pace=clean_str(row_data[22]),
            max_run_cadence=parse_int(row_data[23]),
            moving_time=parse_time(row_data[24]),
            avg_moving_pace=clean_str(row_data[25]),
            avg_stride_speed_loss=clean_str(row_data[26]),
            avg_stride_speed_loss_percent=clean_str(row_data[27])
        )

@dataclasses.dataclass
class WorkoutSummary:
    total_time: str
    cumulative_time: str
    total_distance: float

    @classmethod
    def from_row(cls, row_data: List[str]) -> 'WorkoutSummary':
        # Remove any empty strings and get the actual data
        data = [x for x in row_data if x.strip()]
        return cls(
            total_time=data[1],
            cumulative_time=data[2],
            total_distance=float(data[3].replace(',', '.'))
        )

def parse_lap_line(line: str) -> Optional[Dict]:
    """Parse a single lap line from the Norwegian format."""
    parts = line.split('\t')
    if len(parts) < 21:  # We need at least these many columns for calories
        return None
        
    try:
        lap_num = int(parts[0])
        time = parts[1]
        total_time = parts[2]
        distance = float(parts[3].replace(',', '.')) if parts[3] != '--' else 0
        avg_pace = parts[4]
        avg_hr = int(parts[6]) if parts[6] != '--' else 0
        max_hr = int(parts[7]) if parts[7] != '--' else 0
        total_ascent = int(parts[8]) if parts[8] != '--' else 0
        total_descent = int(parts[9]) if parts[9] != '--' else 0
        total_calories = int(parts[20]) if parts[20] != '--' else 0
        
        return {
            'lap_number': lap_num,
            'time': time,
            'total_time': total_time,
            'distance': distance,
            'avg_pace': avg_pace,
            'avg_hr': avg_hr,
            'max_hr': max_hr,
            'total_ascent': total_ascent,
            'total_descent': total_descent,
            'total_calories': total_calories
        }
    except (ValueError, IndexError):
        return None

def parse_workout(workout_text: str) -> Tuple[List[Dict], Dict]:
    """Parse the entire workout including summary and lap data."""
    lines = workout_text.strip().split('\n')
    
    laps = []
    summary = {}
    in_laps = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Runder'):
            in_laps = True
            continue
            
        if in_laps:
            if line.startswith('Sammendrag'):
                in_laps = False
                continue
                
            lap_data = parse_lap_line(line)
            if lap_data:
                laps.append(lap_data)
                
        # Parse summary data
        if not in_laps:  # Only parse summary after we're done with laps
            if 'km' in line and not line.startswith('Runder'):
                try:
                    distance = float(line.split()[0])
                    summary['distance_km'] = distance
                except (ValueError, IndexError):
                    pass
                    
            elif 'Totale kalorier' in line:
                try:
                    calories = int(line.split()[0])
                    summary['total_calories'] = calories
                except (ValueError, IndexError):
                    pass
                    
            elif 'Gjennomsnittlig puls' in line:
                try:
                    hr = int(line.split()[0])
                    summary['avg_hr'] = hr
                except (ValueError, IndexError):
                    pass
                    
            elif 'Makspuls' in line:
                try:
                    hr = int(line.split()[0])
                    summary['max_hr'] = hr
                except (ValueError, IndexError):
                    pass
                    
            elif 'Total stigning' in line:
                try:
                    ascent = float(line.split()[0])
                    summary['total_ascent_m'] = ascent
                except (ValueError, IndexError):
                    pass
                    
    # If we have laps but no summary, calculate from laps
    if laps and not summary:
        summary['distance_km'] = sum(lap['distance'] for lap in laps)
        summary['total_calories'] = sum(lap['total_calories'] for lap in laps)
        summary['avg_hr'] = sum(lap['avg_hr'] for lap in laps) / len(laps) if any(lap['avg_hr'] for lap in laps) else 0
        summary['max_hr'] = max(lap['max_hr'] for lap in laps) if any(lap['max_hr'] for lap in laps) else 0
        summary['total_ascent_m'] = sum(lap['total_ascent'] for lap in laps)
                
    return laps, summary

def format_workout_summary(laps: List[WorkoutLap], summary: WorkoutSummary) -> str:
    output = []
    
    # Header
    output.append("🏃‍♂️ Workout Summary")
    output.append("=" * 50)
    output.append(f"Total Time: {summary.total_time}")
    output.append(f"Total Distance: {summary.total_distance:.2f} km")
    output.append(f"Number of Laps: {len(laps)}")
    
    # Calculate some additional statistics
    total_calories = sum(lap['total_calories'] for lap in laps)
    total_ascent = sum(lap['total_ascent'] for lap in laps)
    total_descent = sum(lap['total_descent'] for lap in laps)
    avg_hr = sum((lap['avg_hr'] or 0) for lap in laps if lap['avg_hr'] is not None) / len([lap for lap in laps if lap['avg_hr'] is not None])
    max_hr = max((lap['max_hr'] or 0) for lap in laps)
    
    output.append(f"Total Calories: {total_calories}")
    output.append(f"Total Ascent: {total_ascent}m")
    output.append(f"Total Descent: {total_descent}m")
    output.append(f"Average Heart Rate: {avg_hr:.0f} bpm")
    output.append(f"Maximum Heart Rate: {max_hr} bpm")
    
    # Lap Details
    output.append("\nLap Details")
    output.append("=" * 50)
    
    for lap in laps:
        output.append(f"\nLap {lap['lap_number']}:")
        output.append(f"  Time: {lap['time']}")
        output.append(f"  Distance: {lap['distance']:.2f} km")
        output.append(f"  Pace: {lap['avg_pace']}")
        if lap['avg_hr']:
            output.append(f"  Avg HR: {lap['avg_hr']} bpm")
        if lap['avg_run_cadence']:
            output.append(f"  Cadence: {lap['avg_run_cadence']} spm")
        if lap['total_ascent'] > 0 or lap['total_descent'] > 0:
            output.append(f"  Elevation: +{lap['total_ascent']}m/-{lap['total_descent']}m")
    
    return "\n".join(output)

def main():
    # Get input from user
    print("Please paste your workout data (press Enter twice when finished):")
    workout_data = []
    while True:
        line = input()
        if line.strip() == "" and workout_data:  # Empty line and we have some data
            break
        workout_data.append(line)
    
    workout_text = "\n".join(workout_data)
    
    # Parse the workout data
    laps, summary = parse_workout(workout_text)
    
    # Format and print the summary
    print("\n" + format_workout_summary(laps, summary))

if __name__ == "__main__":
    main() 