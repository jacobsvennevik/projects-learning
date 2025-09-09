import sqlite3
import json
from norwegian_workout_parser import parse_workout
from datetime import datetime

def store_workout(workout_text: str, conn: sqlite3.Connection) -> None:
    """Parse and store a workout in the database."""
    cursor = conn.cursor()
    
    # Parse the workout
    laps, summary = parse_workout(workout_text)
    
    if not laps or not summary:
        print("Failed to parse workout data")
        return
        
    # Convert laps to JSON string for storage
    laps_json = json.dumps(laps)
    
    # Insert the workout data
    cursor.execute('''
    INSERT INTO activities (
        date,
        distance_km,
        time,
        avg_pace,
        total_ascent_m,
        total_calories,
        avg_hr,
        max_hr,
        laps,
        raw_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        summary.get('distance_km', 0),
        summary.get('time', ''),
        summary.get('avg_pace', ''),
        summary.get('total_ascent_m', 0),
        summary.get('total_calories', 0),
        summary.get('avg_hr', 0),
        summary.get('max_hr', 0),
        laps_json,
        workout_text
    ))
    
    conn.commit()
    print("Workout stored successfully!")

def main():
    # Connect to the database
    conn = sqlite3.connect('garmin_activities.db')
    cursor = conn.cursor()
    
    # Create the activities table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS activities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        distance_km REAL,
        time TEXT,
        avg_pace TEXT,
        total_ascent_m INTEGER,
        total_calories INTEGER,
        avg_hr INTEGER,
        max_hr INTEGER,
        laps TEXT,
        time_in_zones TEXT,
        raw_json TEXT
    )
    ''')
    conn.commit()
    
    # Your hill sprint workout data
    workout_text = """Runder 	Tid	Samlet tid	Distanse	Gjennomsnittlig tempo	Gjennomsnittlig GAP	Gjennomsnittlig puls	Makspuls	Total stigning	Totalt fall	Gjennomsnittlig kraft	Gjennomsnittlig W/kg	Maksimal kraft	Maks. W/kg	Gjennomsnittlig frekvens for løping	Gjennomsnittlig tid med bakkekontakt	Gjennomsnittlig balanse for tid med bakkekontakt	Gjennomsnittlig skrittlengde	Gjennomsnittlig vertikal oscillasjon	Gjennomsnittlig vertikalt forholdstall	Totale kalorier	Gjennomsnittlig temperatur	Beste tempo	Maksimal frekvens for løping	Tid i bevegelse	Gjennomsnittlig bevegelsestempo	Gj.snittstap av trinnhastighet	Gj.snittstap av trinnhastighet i prosent
1	5:43.9	5:43.9	1.00	5:44	--	127	141	7	16	--	--	--	--	152	--	--	1.15	--	--	64	29.0	5:15	162	5:43	5:43	--	--
2	5:16.3	11:00	1.00	5:16	--	131	142	0	21	--	--	--	--	157	--	--	1.21	--	--	62	26.0	4:58	163	5:16.3	5:16	--	--
3	5:33.4	16:34	1.00	5:33	--	142	149	8	8	--	--	--	--	157	--	--	1.15	--	--	72	23.0	4:56	161	5:33.4	5:33	--	--
4	1:26.3	18:00	0.31	4:36	--	125	135	7	15	--	--	--	--	131	--	--	1.66	--	--	13	24.0	2:28	196	1:21	4:19	--	--
5	1:18.7	19:19	0.34	3:51	--	154	170	11	0	--	--	--	--	169	--	--	1.53	--	--	18	24.0	3:51	184	1:18.7	3:51	--	--
6	2:57.4	22:16	0.39	7:35	--	139	171	3	16	--	--	--	--	124	--	--	1.06	--	--	33	23.0	3:56	218	2:57	7:34	--	--
7	1:17.4	23:33	0.34	3:47	--	153	173	18	5	--	--	--	--	169	--	--	1.56	--	--	17	23.0	3:35	186	1:17.4	3:47	--	--
8	2:59.9	26:33	0.39	7:40	--	145	174	2	15	--	--	--	--	127	--	--	1.03	--	--	37	23.0	3:47	220	2:39	6:46	--	--
9	1:16.5	27:50	0.34	3:48	--	156	176	17	4	--	--	--	--	170	--	--	1.54	--	--	17	23.0	3:37	188	1:14	3:41	--	--
10	2:40.7	30:30	0.38	7:01	--	148	176	2	15	--	--	--	--	134	--	--	1.07	--	--	34	22.0	3:40	213	2:40.7	7:01	--	--
11	1:17.1	31:48	0.34	3:46	--	160	177	14	0	--	--	--	--	179	--	--	1.48	--	--	18	23.0	3:41	192	1:17	3:46	--	--
12	2:55.3	34:43	0.38	7:37	--	144	178	1	13	--	--	--	--	125	--	--	1.05	--	--	34	22.0	3:45	188	2:55	7:37	--	--
13	1:18.3	36:01	0.34	3:53	--	158	176	22	3	--	--	--	--	166	--	--	1.55	--	--	17	23.0	3:29	186	1:18	3:52	--	--
14	2:45.6	38:47	0.39	7:08	--	146	177	0	19	--	--	--	--	134	--	--	1.04	--	--	32	23.0	3:53	184	2:45.6	7:08	--	--
15	1:16.6	40:03	0.34	3:43	--	159	178	12	0	--	--	--	--	175	--	--	1.54	--	--	17	23.0	3:31	188	1:16.6	3:43	--	--
16	2:57.4	43:01	0.39	7:33	--	145	178	3	16	--	--	--	--	128	--	--	1.03	--	--	33	23.0	3:47	186	2:55	7:27	--	--
17	1:17.0	44:18	0.33	3:50	--	159	177	16	2	--	--	--	--	174	--	--	1.50	--	--	17	23.0	3:37	188	1:17.0	3:50	--	--
18	3:19.1	47:37	0.40	8:21	--	145	177	2	14	--	--	--	--	120	--	--	1.00	--	--	35	23.0	3:53	186	2:58	7:28	--	--
19	1:16.9	48:54	0.34	3:44	--	161	180	13	1	--	--	--	--	181	--	--	1.48	--	--	17	23.0	3:25	196	1:16.9	3:44	--	--
20	3:46.7	52:40	0.38	9:49	--	140	180	2	14	--	--	--	--	107	--	--	0.96	--	--	35	22.0	3:42	188	3:14	8:24	--	--
21	1:14.6	53:55	0.33	3:44	--	160	179	11	0	--	--	--	--	173	--	--	1.54	--	--	16	23.0	3:30	191	1:13	3:39	--	--
22	3:40.5	57:36	0.44	8:18	--	141	181	2	14	--	--	--	--	129	--	--	0.93	--	--	35	22.0	3:45	196	3:40.5	8:18	--	--
23	1:15.8	58:51	0.34	3:43	--	161	181	12	0	--	--	--	--	180	--	--	1.49	--	--	17	22.0	3:44	207	1:15.8	3:43	--	--
24	3:18.0	1:02:09	0.38	8:35	--	143	181	3	15	--	--	--	--	137	--	--	0.85	--	--	35	22.0	3:45	186	3:18	8:35	--	--
25	1:17.3	1:03:27	0.33	3:52	--	160	180	12	0	--	--	--	--	144	--	--	1.79	--	--	17	22.0	3:38	191	1:17	3:51	--	--
26	6:28.0	1:09:55	1.00	6:28	--	146	180	10	4	--	--	--	--	154	--	--	1.00	--	--	70	22.0	3:51	184	6:28.0	6:28	--	--
27	1:58.7	1:11:53	0.35	5:37	--	143	148	1	1	--	--	--	--	156	--	--	1.14	--	--	19	22.0	4:55	229	1:58.7	5:37	--	--
Sammendrag	1:11:53	1:11:53	12.32	5:50		1"""
    
    # Store the workout
    store_workout(workout_text, conn)
    
    # Close the connection
    conn.close()

if __name__ == "__main__":
    main() 