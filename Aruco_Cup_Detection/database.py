import sqlite3
from datetime import datetime

DATABASE = 'database.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS violations (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        frame_index INTEGER,
                        cup_count INTEGER,
                        aruco_count INTEGER,
                        aruco_ids TEXT)''')
    conn.commit()
    conn.close()

from sqlite3 import Error

def log_violation(frame_index, cup_count, aruco_count, aruco_ids):
    # Skip logging if aruco_ids contain specific values
    if any(x in aruco_ids for x in [17,10, 37, 38, 62, 80]):
        return

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Convert the list of aruco_ids to a comma-separated string
    aruco_ids_str = ','.join(map(str, aruco_ids))

    try:
        # Check the last recorded violation
        cursor.execute("SELECT * FROM violations ORDER BY id DESC LIMIT 1")
        last_violation = cursor.fetchone()

        # If there was a previous violation, check the ArUco IDs
        if last_violation:
            last_aruco_ids = last_violation[5]
            if last_aruco_ids:
                last_aruco_ids = list(map(int, last_aruco_ids.split(',')))
            else:
                last_aruco_ids = []

            # Only log if there are new ArUco IDs
            if any(aruco_id not in last_aruco_ids for aruco_id in aruco_ids):
                current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute("INSERT INTO violations (timestamp, frame_index, cup_count, aruco_count, aruco_ids) VALUES (?, ?, ?, ?, ?)",
                               (current_timestamp, frame_index, cup_count, aruco_count, aruco_ids_str))
                conn.commit()
        else:
            # No previous violations, log this one
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute("INSERT INTO violations (timestamp, frame_index, cup_count, aruco_count, aruco_ids) VALUES (?, ?, ?, ?, ?)",
                           (timestamp, frame_index, cup_count, aruco_count, aruco_ids_str))
            conn.commit()

        # Automatically remove entry if it has an aruco_count of 0
        cursor.execute("DELETE FROM violations WHERE aruco_count = 0")
        conn.commit()

    except Error as e:
        print(f"Error logging violation: {e}")
    finally:
        conn.close()

def get_analytics():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM violations")
    data = cursor.fetchall()
    conn.close()
    return data

def clear_table():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM violations")
    conn.commit()
    conn.close()

