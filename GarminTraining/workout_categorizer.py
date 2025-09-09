import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import re
import json

class WorkoutType(Enum):
    CALM_RUN = "Calm Run"
    THRESHOLD_RUN = "Threshold Run"
    HILL_SPRINTS = "Hill Sprints"
    UNKNOWN = "Unknown"

@dataclass
class IntervalStats:
    count: int
    avg_distance: float
    avg_time: str
    avg_pace: str
    avg_hr: float
    max_hr: int
    total_distance: float
    recovery_time: str
    avg_recovery_time: str
    warmup_distance: float
    warmup_time: str
    warmup_avg_hr: float
    avg_elevation_gain: float = 0
    total_elevation_gain: float = 0
    avg_elevation_gain_percent: float = 0

@dataclass
class WorkoutCategory:
    workout_id: int
    type: WorkoutType
    distance_km: float
    avg_hr: int
    avg_pace: str
    time_in_zone2_percent: float
    total_time: str
    date: str
    interval_stats: Optional[IntervalStats] = None

def parse_time_to_seconds(time_str: str) -> float:
    """Convert time string (MM:SS or HH:MM:SS) to seconds"""
    if not time_str or time_str == '--':
        return 0
    parts = time_str.split(':')
    if len(parts) == 2:  # MM:SS
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:  # HH:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return 0

def format_seconds_to_pace(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    if seconds == 0:
        return '--'
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}:{remaining_seconds:02d}"

def normalize_interval_distance(distance: float) -> float:
    """
    Normalize interval distance to either 400m or 1000m based on closest match.
    This helps correct for GPS inaccuracies and gives us the intended interval distance.
    """
    if abs(distance - 0.4) <= 0.1:  # If close to 400m
        return 0.4
    elif abs(distance - 1.0) <= 0.1:  # If close to 1000m
        return 1.0
    return distance

def analyze_intervals(laps: List[tuple]) -> Optional[IntervalStats]:
    """
    Analyze lap data to identify and characterize intervals.
    Returns None if the pattern doesn't match threshold training.
    """
    if not laps or len(laps) < 10:  # Need at least 5 intervals (10 laps including recovery)
        return None
    
    # First, identify warmup laps (typically in zones 1-2, lower heart rate)
    warmup_laps = []
    main_workout_laps = []
    in_warmup = True
    
    for lap in laps:
        # Unpack the lap data - format from database is:
        # id, workout_id, lap_number, time, total_time, distance_km, avg_pace, avg_gap, avg_hr, max_hr, ...
        lap_number = lap[2]
        time = lap[3]
        distance = float(lap[5]) if lap[5] != '--' else 0
        avg_hr = float(lap[8]) if lap[8] != '--' else 0
        max_hr = float(lap[9]) if lap[9] != '--' else 0
        
        # Consider it still warmup if HR is low (zones 1-2, typically under 136 bpm)
        # or if we're in the first two laps
        if in_warmup and (int(lap_number) <= 2 or (avg_hr and avg_hr <= 136)):
            warmup_laps.append((lap_number, time, distance, avg_hr, max_hr))
        else:
            in_warmup = False  # Once we hit higher HR, warmup is done
            main_workout_laps.append((lap_number, time, distance, avg_hr, max_hr))
    
    # Now analyze the main workout laps
    intervals = []
    recoveries = []
    potential_intervals = []
    
    # First pass - identify potential intervals and calculate their average pace
    for lap in main_workout_laps:
        lap_number, time, distance, avg_hr, max_hr = lap
        # Convert time to seconds for calculations
        time_seconds = parse_time_to_seconds(time)
        
        # Skip very short laps (less than 30m) as they might be GPS errors
        if distance < 0.03:
            continue
            
        # Calculate pace in seconds per km
        pace = time_seconds / distance if distance > 0 else float('inf')
            
        # Potential interval if distance is close to 1000m or 400m
        if abs(distance - 1.0) <= 0.1 or abs(distance - 0.4) <= 0.05:
            potential_intervals.append((lap_number, time, distance, avg_hr, max_hr, pace))
    
    if not potential_intervals:
        return None
        
    # Calculate average interval pace
    avg_interval_pace = sum(i[5] for i in potential_intervals) / len(potential_intervals)
    
    # Second pass - categorize laps as intervals or recoveries based on pace
    for lap in main_workout_laps:
        lap_number, time, distance, avg_hr, max_hr = lap
        time_seconds = parse_time_to_seconds(time)
        
        # Skip very short laps (less than 30m) as they might be GPS errors
        if distance < 0.03:
            continue
            
        # Calculate pace in seconds per km
        pace = time_seconds / distance if distance > 0 else float('inf')
        
        # If the pace is significantly slower (>50% slower) than the average interval pace,
        # consider it a recovery, otherwise check if it's an interval
        if pace > avg_interval_pace * 1.5:  # Recovery is at least 50% slower
            recoveries.append((distance, time_seconds, avg_hr, max_hr))
        elif abs(distance - 1.0) <= 0.1 or abs(distance - 0.4) <= 0.05:
            # Normalize the distance to exactly 400m or 1000m
            normalized_distance = normalize_interval_distance(distance)
            # Adjust pace calculation based on normalized distance
            normalized_time = normalized_distance * (time_seconds / distance)  # Keep the same pace
            intervals.append((normalized_distance, normalized_time, avg_hr, max_hr))
    
    # Check if we have a valid interval pattern
    if len(intervals) < 5:  # Need at least 5 intervals
        return None
        
    # Calculate interval statistics
    total_interval_distance = sum(i[0] for i in intervals)
    avg_interval_distance = total_interval_distance / len(intervals)
    
    # If average distance is neither close to 1000m nor 400m, it's not a threshold run
    if not (abs(avg_interval_distance - 1.0) <= 0.1 or abs(avg_interval_distance - 0.4) <= 0.05):
        return None
    
    # Calculate averages for valid intervals
    avg_time_seconds = sum(i[1] for i in intervals) / len(intervals)
    avg_hr = sum(i[2] for i in intervals) / len(intervals)
    max_hr = max(i[3] for i in intervals)
    
    # Calculate average pace (min/km) from normalized time and distance
    avg_pace_seconds = avg_time_seconds / avg_interval_distance
    
    # Calculate recovery time statistics
    total_recovery_seconds = sum(r[1] for r in recoveries)
    avg_recovery_seconds = total_recovery_seconds / len(recoveries) if recoveries else 0
    
    # Calculate warmup stats
    warmup_distance = sum(lap[2] for lap in warmup_laps)
    warmup_time = sum(parse_time_to_seconds(lap[1]) for lap in warmup_laps)
    warmup_avg_hr = sum(lap[3] for lap in warmup_laps if lap[3]) / len(warmup_laps) if warmup_laps else 0
    
    return IntervalStats(
        count=len(intervals),
        avg_distance=round(avg_interval_distance * 1000) / 1000,  # Round to nearest meter
        avg_time=format_seconds_to_pace(avg_time_seconds),
        avg_pace=format_seconds_to_pace(avg_pace_seconds),
        avg_hr=avg_hr,
        max_hr=max_hr,
        total_distance=round(total_interval_distance * 1000) / 1000,  # Round to nearest meter
        recovery_time=format_seconds_to_pace(total_recovery_seconds),
        avg_recovery_time=format_seconds_to_pace(avg_recovery_seconds),
        warmup_distance=warmup_distance,
        warmup_time=format_seconds_to_pace(warmup_time),
        warmup_avg_hr=warmup_avg_hr
    )

def analyze_hill_sprints(laps: List[tuple]) -> Optional[IntervalStats]:
    """
    Analyze lap data to identify hill sprint intervals.
    Returns None if the pattern doesn't match hill sprint training.
    """
    if not laps or len(laps) < 6:  # Need at least 3 intervals (with recoveries)
        return None
    
    # First, identify warmup laps (typically in zones 1-2, lower heart rate)
    warmup_laps = []
    main_workout_laps = []
    in_warmup = True
    
    for lap in laps:
        # Unpack the lap data - format from database is:
        # id, workout_id, lap_number, time, total_time, distance_km, avg_pace, avg_gap, avg_hr, max_hr, total_ascent, total_descent, ...
        lap_number = lap[2]
        time = lap[3]
        distance = float(lap[5]) if lap[5] != '--' else 0
        avg_pace = lap[6]
        avg_hr = float(lap[8]) if lap[8] != '--' else 0
        max_hr = float(lap[9]) if lap[9] != '--' else 0
        total_ascent = float(lap[10]) if lap[10] != '--' else 0
        total_descent = float(lap[11]) if lap[11] != '--' else 0
        
        # Convert pace to seconds for comparison
        pace_seconds = parse_time_to_seconds(avg_pace)
        
        # Consider it warmup if:
        # 1. We're still in warmup phase AND
        # 2. Heart rate is in zones 1-2 (≤ 136 bpm) OR
        # 3. Pace is easy (> 5:00/km)
        if in_warmup and (avg_hr <= 136 or pace_seconds > parse_time_to_seconds("5:00")):
            warmup_laps.append((lap_number, time, distance, avg_hr, max_hr))
        else:
            in_warmup = False  # Once we hit higher intensity, warmup is done
            main_workout_laps.append((lap_number, time, distance, avg_hr, max_hr, total_ascent, total_descent))
    
    # Now analyze the main workout laps
    intervals = []
    recoveries = []
    
    # First pass - identify hill sprints based on:
    # 1. Distance (~0.31-0.35km)
    # 2. Elevation gain (>4% grade OR >10m absolute gain)
    # 3. Higher intensity (pace under 5:00/km)
    for lap in main_workout_laps:
        lap_number, time, distance, avg_hr, max_hr, total_ascent, total_descent = lap
        time_seconds = parse_time_to_seconds(time)
        
        # Skip very short laps (less than 30m) as they might be GPS errors
        if distance < 0.03:
            continue
            
        # Calculate pace in seconds per km
        pace = time_seconds / distance if distance > 0 else float('inf')
        
        # Calculate elevation gain percentage
        elevation_gain_percent = (total_ascent / (distance * 1000)) * 100 if distance > 0 else 0
        
        # Hill sprint characteristics:
        # 1. Distance around 0.33km (±0.03km)
        # 2. Significant elevation gain (>4% grade OR >10m absolute)
        # 3. Higher intensity (pace under 5:00/km)
        if (0.31 <= distance <= 0.35 and 
            (elevation_gain_percent > 4 or total_ascent > 10) and 
            pace < parse_time_to_seconds("5:00")):
            intervals.append((distance, time_seconds, avg_hr, max_hr, total_ascent, elevation_gain_percent))
        # Recovery characteristics:
        # 1. Longer distance (>0.35km)
        # 2. Net descent or minimal elevation gain
        # 3. Slower pace
        elif (distance > 0.35 and 
              total_descent >= total_ascent):
            recoveries.append((distance, time_seconds, avg_hr, max_hr))
    
    # Check if we have enough hill sprints
    if len(intervals) < 3:  # Need at least 3 hill sprints
        return None
        
    # Calculate interval statistics
    total_interval_distance = sum(i[0] for i in intervals)
    avg_interval_distance = total_interval_distance / len(intervals)
    avg_time_seconds = sum(i[1] for i in intervals) / len(intervals)
    avg_hr = sum(i[2] for i in intervals) / len(intervals)
    max_hr = max(i[3] for i in intervals)
    
    # Calculate elevation statistics
    total_elevation_gain = sum(i[4] for i in intervals)
    avg_elevation_gain = total_elevation_gain / len(intervals)
    avg_elevation_gain_percent = sum(i[5] for i in intervals) / len(intervals)
    
    # Calculate average pace (min/km) from time and distance
    avg_pace_seconds = avg_time_seconds / avg_interval_distance
    
    # Calculate recovery time statistics
    total_recovery_seconds = sum(r[1] for r in recoveries)
    avg_recovery_seconds = total_recovery_seconds / len(recoveries) if recoveries else 0
    
    # Calculate warmup stats
    warmup_distance = sum(lap[2] for lap in warmup_laps)
    warmup_time = sum(parse_time_to_seconds(lap[1]) for lap in warmup_laps)
    warmup_avg_hr = sum(lap[3] for lap in warmup_laps if lap[3]) / len(warmup_laps) if warmup_laps else 0
    
    return IntervalStats(
        count=len(intervals),
        avg_distance=round(avg_interval_distance * 1000) / 1000,  # Round to nearest meter
        avg_time=format_seconds_to_pace(avg_time_seconds),
        avg_pace=format_seconds_to_pace(avg_pace_seconds),
        avg_hr=avg_hr,
        max_hr=max_hr,
        total_distance=round(total_interval_distance * 1000) / 1000,  # Round to nearest meter
        recovery_time=format_seconds_to_pace(total_recovery_seconds),
        avg_recovery_time=format_seconds_to_pace(avg_recovery_seconds),
        warmup_distance=warmup_distance,
        warmup_time=format_seconds_to_pace(warmup_time),
        warmup_avg_hr=warmup_avg_hr,
        avg_elevation_gain=round(avg_elevation_gain),
        total_elevation_gain=round(total_elevation_gain),
        avg_elevation_gain_percent=round(avg_elevation_gain_percent, 1)
    )

def categorize_workout(time_in_zone2_percent: float, interval_stats: Optional[IntervalStats], hill_stats: Optional[IntervalStats]) -> WorkoutType:
    """
    Categorize a workout based on time spent in different heart rate zones and interval patterns
    """
    if hill_stats:
        # If we detected valid hill sprint intervals with significant elevation gain
        return WorkoutType.HILL_SPRINTS
    elif interval_stats:
        # If we detected a valid interval pattern, it's a threshold run
        return WorkoutType.THRESHOLD_RUN
    elif time_in_zone2_percent >= 90.0:
        return WorkoutType.CALM_RUN
    return WorkoutType.UNKNOWN

def format_workout_category(workout: WorkoutCategory) -> str:
    """
    Format a workout category for display
    """
    base_info = f"""
Workout on {workout.date}
Type: {workout.type.value}
Distance: {workout.distance_km:.2f} km
Duration: {workout.total_time}
Average HR: {workout.avg_hr} bpm
Average Pace: {workout.avg_pace}
Time in Zone 2: {workout.time_in_zone2_percent:.1f}%"""

    if workout.type == WorkoutType.HILL_SPRINTS and workout.interval_stats:
        interval_info = f"""
=== Warmup ===
Distance: {workout.interval_stats.warmup_distance:.2f}km
Duration: {workout.interval_stats.warmup_time}
Average HR: {workout.interval_stats.warmup_avg_hr:.1f} bpm

=== Hill Sprint Details ===
Number of Sprints: {workout.interval_stats.count}
Average Sprint Distance: {workout.interval_stats.avg_distance:.2f}km
Average Sprint Time: {workout.interval_stats.avg_time}
Average Sprint Pace: {workout.interval_stats.avg_pace}/km
Average Sprint HR: {workout.interval_stats.avg_hr:.1f}
Max Sprint HR: {workout.interval_stats.max_hr}
Total Sprint Distance: {workout.interval_stats.total_distance:.2f}km
Average Elevation Gain per Sprint: {workout.interval_stats.avg_elevation_gain}m
Total Elevation Gain: {workout.interval_stats.total_elevation_gain}m
Average Elevation Gain Percentage: {workout.interval_stats.avg_elevation_gain_percent}%
Average Recovery Time: {workout.interval_stats.avg_recovery_time}"""
        return base_info + interval_info
    elif workout.type == WorkoutType.THRESHOLD_RUN and workout.interval_stats:
        interval_info = f"""
=== Warmup ===
Distance: {workout.interval_stats.warmup_distance:.2f}km
Duration: {workout.interval_stats.warmup_time}
Average HR: {workout.interval_stats.warmup_avg_hr:.1f} bpm

=== Interval Details ===
Number of Intervals: {workout.interval_stats.count}
Interval Distance: {workout.interval_stats.avg_distance:.2f}km
Average Interval Time: {workout.interval_stats.avg_time}
Average Interval Pace: {workout.interval_stats.avg_pace}/km
Average Interval HR: {workout.interval_stats.avg_hr:.1f}
Max Interval HR: {workout.interval_stats.max_hr}
Total Interval Distance: {workout.interval_stats.total_distance:.2f}km
Average Recovery Time: {workout.interval_stats.avg_recovery_time}"""
        return base_info + interval_info
    
    return base_info

def get_workout_data(db_path: str = 'workout_summaries.db') -> List[WorkoutCategory]:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Get all workouts with their heart rate zone data
    c.execute('''
        SELECT 
            w.id,
            w.distance_km,
            w.avg_hr,
            w.avg_pace,
            w.total_time,
            w.date,
            GROUP_CONCAT(hz.zone_number || ':' || hz.percentage)
        FROM workouts w
        LEFT JOIN heart_rate_zones hz ON w.id = hz.workout_id
        GROUP BY w.id
    ''')
    
    workouts = []
    for row in c.fetchall():
        workout_id, distance, avg_hr, avg_pace, total_time, date, zone_data = row
        
        # Get lap data for this workout
        c.execute('''
            SELECT *
            FROM laps
            WHERE workout_id = ?
            ORDER BY lap_number
        ''', (workout_id,))
        laps = c.fetchall()
        
        # Parse zone data
        time_in_zone2 = 0.0
        if zone_data:
            zones = dict(
                map(lambda x: (int(x.split(':')[0]), float(x.split(':')[1])), 
                    zone_data.split(','))
            )
            time_in_zone2 = zones.get(2, 0.0)
        
        # Analyze intervals - try both threshold and hill sprint detection
        interval_stats = analyze_intervals(laps)
        hill_stats = analyze_hill_sprints(laps)
        
        # Categorize workout
        workout_type = categorize_workout(time_in_zone2, interval_stats, hill_stats)
        
        # Use the appropriate stats based on workout type
        stats = hill_stats if workout_type == WorkoutType.HILL_SPRINTS else interval_stats
        
        workouts.append(WorkoutCategory(
            workout_id=workout_id,
            type=workout_type,
            distance_km=distance,
            avg_hr=avg_hr or 0,
            avg_pace=avg_pace or "unknown",
            time_in_zone2_percent=time_in_zone2,
            total_time=total_time or "unknown",
            date=date,
            interval_stats=stats
        ))
    
    conn.close()
    return workouts

def identify_workout_type(laps: List[tuple]) -> str:
    """
    Analyze the workout pattern to identify the type of workout.
    Returns a string describing the workout type.
    """
    if not laps:
        return "Unknown workout (no data)"
        
    # Look for hill sprint pattern
    hill_stats = analyze_hill_sprints(laps)
    if hill_stats:
        return f"Hill Sprint Workout ({hill_stats.count} sprints)"
        
    # Look for threshold pattern
    threshold_stats = analyze_intervals(laps)
    if threshold_stats:
        interval_dist = threshold_stats.avg_distance
        if abs(interval_dist - 0.4) <= 0.05:
            return f"Threshold Run (400m intervals × {threshold_stats.count})"
        elif abs(interval_dist - 1.0) <= 0.1:
            return f"Threshold Run (1000m intervals × {threshold_stats.count})"
        return f"Interval Workout ({interval_dist*1000:.0f}m intervals × {threshold_stats.count})"
    
    # If no specific pattern found, analyze general characteristics
    total_distance = sum(float(lap[5]) for lap in laps if lap[5] != '--')
    avg_hr = sum(float(lap[8]) for lap in laps if lap[8] != '--') / len([lap for lap in laps if lap[8] != '--'])
    
    if avg_hr <= 136:
        return f"Easy/Recovery Run ({total_distance:.2f}km)"
    elif avg_hr <= 156:
        return f"Steady State Run ({total_distance:.2f}km)"
    else:
        return f"Tempo Run ({total_distance:.2f}km)"

def main():
    print("Analyzing workouts...")
    workouts = get_workout_data()
    
    print("\nWorkout Categories:")
    print("=" * 50)
    for workout in workouts:
        print(format_workout_category(workout))

if __name__ == "__main__":
    main() 