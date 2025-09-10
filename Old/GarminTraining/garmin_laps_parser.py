import pandas as pd
import numpy as np
from typing import Optional
import re

__all__ = ['parse_laps', '_to_seconds', '_infer_delimiter']

def _to_seconds(time_str: str) -> Optional[float]:
    """Convert time strings to seconds."""
    try:
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        elif len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None


def _infer_delimiter(line: str) -> str:
    """Infer the delimiter used in the laps table."""
    if '\t' in line:
        return '\t'
    # Use raw string for regex to handle multiple spaces
    return r'\s{2,}'


def parse_laps(table_txt: str) -> pd.DataFrame:
    """
    Parse the Garmin Connect laps table into a pandas DataFrame.
    """
    lines = table_txt.strip().split('\n')
    header = lines[0]
    delimiter = _infer_delimiter(header)
    data = [re.split(delimiter, line.strip()) for line in lines[1:] if not line.startswith('Sammendrag')]

    # Define column names in English snake_case
    columns = [
        'lap', 'time_s', 'cumulative_time_s', 'distance_km',
        'avg_pace_s_per_km', 'avg_hr', 'max_hr', 'total_ascent_m',
        'total_descent_m', 'avg_power', 'avg_w_per_kg', 'max_power',
        'max_w_per_kg', 'avg_cadence_spm', 'avg_ground_contact_time_ms',
        'avg_ground_contact_balance', 'avg_stride_length_m',
        'avg_vertical_oscillation_cm', 'avg_vertical_ratio',
        'total_calories', 'avg_temp_c', 'best_pace_s_per_km',
        'max_cadence_spm', 'moving_time_s', 'avg_moving_pace_s_per_km',
        'avg_step_loss', 'avg_step_loss_percent'
    ]

    # Ensure each row has the correct number of columns
    data = [row for row in data if len(row) == len(columns)]

    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Convert columns to appropriate dtypes
    df['lap'] = df['lap'].astype('int16')
    df['time_s'] = df['time_s'].apply(_to_seconds).astype('float32')
    df['cumulative_time_s'] = df['cumulative_time_s'].apply(_to_seconds).astype('float32')
    df['distance_km'] = df['distance_km'].replace('--', np.nan).astype('float32')
    df['avg_pace_s_per_km'] = df['avg_pace_s_per_km'].apply(lambda x: _to_seconds(x) if x != '--' else np.nan).astype('float32')
    df['avg_hr'] = df['avg_hr'].replace('--', np.nan).astype('float32')
    df['max_hr'] = df['max_hr'].replace('--', np.nan).astype('float32')
    df['total_ascent_m'] = df['total_ascent_m'].replace('--', np.nan).astype('float32')
    df['total_descent_m'] = df['total_descent_m'].replace('--', np.nan).astype('float32')
    df['avg_power'] = df['avg_power'].replace('--', np.nan).astype('float32')
    df['avg_w_per_kg'] = df['avg_w_per_kg'].replace('--', np.nan).astype('float32')
    df['max_power'] = df['max_power'].replace('--', np.nan).astype('float32')
    df['max_w_per_kg'] = df['max_w_per_kg'].replace('--', np.nan).astype('float32')
    df['avg_cadence_spm'] = df['avg_cadence_spm'].replace('--', np.nan).astype('float32')
    df['avg_ground_contact_time_ms'] = df['avg_ground_contact_time_ms'].replace('--', np.nan).astype('float32')
    df['avg_ground_contact_balance'] = df['avg_ground_contact_balance'].replace('--', np.nan).astype('float32')
    df['avg_stride_length_m'] = df['avg_stride_length_m'].replace('--', np.nan).astype('float32')
    df['avg_vertical_oscillation_cm'] = df['avg_vertical_oscillation_cm'].replace('--', np.nan).astype('float32')
    df['avg_vertical_ratio'] = df['avg_vertical_ratio'].replace('--', np.nan).astype('float32')
    df['total_calories'] = df['total_calories'].replace('--', np.nan).astype('float32')
    df['avg_temp_c'] = df['avg_temp_c'].replace('--', np.nan).astype('float32')
    df['best_pace_s_per_km'] = df['best_pace_s_per_km'].apply(lambda x: _to_seconds(x) if x != '--' else np.nan).astype('float32')
    df['max_cadence_spm'] = df['max_cadence_spm'].replace('--', np.nan).astype('float32')
    df['moving_time_s'] = df['moving_time_s'].apply(_to_seconds).astype('float32')
    df['avg_moving_pace_s_per_km'] = df['avg_moving_pace_s_per_km'].apply(lambda x: _to_seconds(x) if x != '--' else np.nan).astype('float32')
    df['avg_step_loss'] = df['avg_step_loss'].replace('--', np.nan).astype('float32')
    df['avg_step_loss_percent'] = df['avg_step_loss_percent'].replace('--', np.nan).astype('float32')

    return df

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python garmin_laps_parser.py laps.txt")
        sys.exit(1)

    file_path = sys.argv[1]
    with open(file_path, 'r') as file:
        table_txt = file.read()

    df = parse_laps(table_txt)
    print(df.head()) 