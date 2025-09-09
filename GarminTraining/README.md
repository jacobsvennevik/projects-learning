# Garmin Training Tracker 🏃‍♂️

A comprehensive Python application for syncing Garmin Connect activities to a local SQLite database with intelligent incremental syncing and detailed progress reporting.

## Features

✅ **Secure Login** - Uses the official garminconnect community SDK (no Selenium required)  
✅ **Incremental Sync** - Only fetches new activities, stops when encountering existing ones  
✅ **Comprehensive Data** - Fetches summary, laps, and heart rate zones for each activity  
✅ **Rich Normalization** - Extracts and calculates key metrics like pace, training effect, etc.  
✅ **Local Storage** - Stores both normalized data and raw JSON in SQLite  
✅ **Progress Reports** - Generate detailed year-to-date statistics and summaries  
✅ **CLI Interface** - Simple command-line interface for sync and reporting  

## Installation

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Sync Activities
```bash
# Interactive login (prompts for credentials)
python tracker.py sync

# Pass credentials as arguments
python tracker.py sync -u your_username -p your_password
```

### Generate Reports
```bash
python tracker.py report
```

## Sample Output

### Sync Output
```
✅ Successfully logged in to Garmin Connect
🔄 Starting activity sync...
📥 Fetching activities (start=0, limit=20)
📊 Fetching details for activity 12345678...
[NEW] 2025-01-15 | 12.95 km | 4:52/km | TE 4.8 / 2.6 → stored ✅
📊 Fetching details for activity 12345679...
[NEW] 2025-01-14 | 8.20 km | 5:12/km | TE 3.2 / 1.8 → stored ✅
[STOP] Encountered existing activityId 12345680 – sync complete.
✅ Sync finished. 2 new activities processed.
```

### Report Output
```
📊 Generating Year-to-Date Report
==================================================
📅 Year 2025 Summary:
   Total Activities: 42
   Total Distance: 387.5 km
   Total Time: 32.4 hours
   Average Pace: 5:02/km

🏃 Recent Activities:
--------------------------------------------------
   2025-01-15 | 12.95 km | 4:52/km | TE 4.8/2.6
   2025-01-14 | 8.20 km | 5:12/km | TE 3.2/1.8
   2025-01-13 | 15.00 km | 4:45/km | TE 5.1/3.2
   ...
==================================================
```

## Database Schema

The application creates a SQLite database (`garmin_activities.db`) with the following structure:

```sql
CREATE TABLE activities (
    id INTEGER PRIMARY KEY,              -- Garmin activity ID
    start_date TEXT NOT NULL,            -- Activity date (YYYY-MM-DD)
    distance_km REAL,                    -- Distance in kilometers
    moving_time_s INTEGER,               -- Moving time in seconds
    elapsed_time_s INTEGER,              -- Total elapsed time in seconds
    avg_pace_s_per_km REAL,             -- Average pace in seconds per km
    aerobic_te REAL,                     -- Aerobic training effect
    anaerobic_te REAL,                   -- Anaerobic training effect
    avg_hr INTEGER,                      -- Average heart rate
    max_hr INTEGER,                      -- Maximum heart rate
    ascent_m REAL,                       -- Total elevation gain in meters
    descent_m REAL,                      -- Total elevation loss in meters
    avg_stride_len_m REAL,               -- Average stride length in meters
    avg_cadence_spm REAL,                -- Average cadence (steps per minute)
    avg_temp_c REAL,                     -- Average temperature in Celsius
    raw_json TEXT NOT NULL,              -- Complete raw activity data as JSON
    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## How It Works

1. **Login**: Authenticates with Garmin Connect using your credentials
2. **Incremental Paging**: Fetches activities in reverse chronological order (newest first)
3. **Smart Stopping**: Stops syncing when it encounters an activity that already exists in the database
4. **Comprehensive Fetching**: For each new activity, fetches three detailed endpoints:
   - `/activity-service/activity/{id}` (summary data)
   - `/activity-service/activity/{id}/laps` (lap-by-lap data)
   - `/activity-service/activity/{id}/zones/heartRate` (heart rate zones)
5. **Data Normalization**: Extracts and calculates key metrics from the raw data
6. **Storage**: Stores both normalized metrics and complete raw JSON for future analysis

## Extracted Metrics

The application normalizes and stores the following key metrics:

- **Distance** (km)
- **Moving Time** (seconds)
- **Elapsed Time** (seconds)
- **Average Pace** (seconds per km)
- **Training Effect** (aerobic and anaerobic)
- **Heart Rate** (average and maximum)
- **Elevation** (ascent and descent in meters)
- **Stride Length** (average in meters)
- **Cadence** (steps per minute)
- **Temperature** (average in Celsius)

## Security Notes

- Credentials are only used for authentication and are not stored
- All data is stored locally in SQLite
- Uses the official garminconnect SDK for secure API access
- No web scraping or browser automation required

## Requirements

- Python 3.6+
- Active Garmin Connect account
- Internet connection for syncing

## Error Handling

The application includes comprehensive error handling:
- Connection failures are gracefully handled
- Invalid activities are skipped
- Database operations are transactional
- Clear error messages are provided

## Contributing

Feel free to submit issues and pull requests. This is a community-driven project designed to help athletes track and analyze their training data.

## License

This project is open source and available under the MIT License. 