import sqlite3
import json
from workout_categorizer import analyze_hill_sprints, identify_workout_type

def test_hill_sprint_detection():
    # Connect to the database
    conn = sqlite3.connect('garmin_activities.db')
    cursor = conn.cursor()
    
    # Get the latest workout
    cursor.execute('SELECT id, laps FROM activities ORDER BY id DESC LIMIT 1')
    workout_id, laps_json = cursor.fetchone()
    
    # Parse the laps JSON
    laps = json.loads(laps_json)
    
    # Convert laps to the format expected by analyze_hill_sprints
    formatted_laps = []
    for lap in laps:
        formatted_lap = (
            None,  # id
            workout_id,  # workout_id
            lap['lap_number'],  # lap_number
            lap['time'],  # time
            lap['total_time'],  # total_time
            lap['distance'],  # distance_km
            lap['avg_pace'],  # avg_pace
            '--',  # avg_gap
            lap['avg_hr'],  # avg_hr
            lap['max_hr'],  # max_hr
            lap['total_ascent'],  # total_ascent
            lap['total_descent']  # total_descent
        )
        formatted_laps.append(formatted_lap)
    
    # First identify the workout type
    workout_type = identify_workout_type(formatted_laps)
    print(f"\nWorkout Type: {workout_type}")
    
    # Analyze the workout
    result = analyze_hill_sprints(formatted_laps)
    
    if result:
        print("\nDetailed Analysis:")
        print(f"Number of hill sprints: {result.count}")
        print(f"Average sprint distance: {result.avg_distance:.2f}km")
        print(f"Average sprint time: {result.avg_time}")
        print(f"Average sprint pace: {result.avg_pace}")
        print(f"Average heart rate: {result.avg_hr:.1f} bpm")
        print(f"Max heart rate: {result.max_hr} bpm")
        print(f"Total sprint distance: {result.total_distance:.2f}km")
        print(f"Total recovery time: {result.recovery_time}")
        print(f"Average recovery time: {result.avg_recovery_time}")
        print(f"Average elevation gain per sprint: {result.avg_elevation_gain}m")
        print(f"Average elevation gain percentage: {result.avg_elevation_gain_percent}%")
        print(f"Total elevation gain: {result.total_elevation_gain}m")
        print("\nWarmup stats:")
        print(f"Distance: {result.warmup_distance:.2f}km")
        print(f"Time: {result.warmup_time}")
        print(f"Average HR: {result.warmup_avg_hr:.1f} bpm")
    else:
        print("\nNot a hill sprint workout")
        
    conn.close()

if __name__ == "__main__":
    test_hill_sprint_detection() 