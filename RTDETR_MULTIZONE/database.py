import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    # Create the table if it does not exist
    c.execute('''CREATE TABLE IF NOT EXISTS movements
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  region INTEGER, 
                  timestamp TEXT, 
                  direction TEXT, 
                  count INTEGER, 
                  class_name TEXT)''')
    # Check if the 'region' column exists
    c.execute("PRAGMA table_info(movements)")
    columns = [col[1] for col in c.fetchall()]
    if 'region' not in columns:
        # Alter the table to add the 'region' column if it doesn't exist
        c.execute("ALTER TABLE movements ADD COLUMN region INTEGER")
    conn.commit()
    conn.close()

def log_movement(region, direction, count, class_name):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('INSERT INTO movements (region, timestamp, direction, count, class_name) VALUES (?, ?, ?, ?, ?)',
              (region, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), direction, count, class_name))
    conn.commit()
    conn.close()

def get_analytics():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('SELECT region, timestamp, direction, count, class_name FROM movements ORDER BY timestamp DESC')
    data = c.fetchall()
    conn.close()
    return data

def clear_table():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('DELETE FROM movements')
    conn.commit()
    conn.close()
