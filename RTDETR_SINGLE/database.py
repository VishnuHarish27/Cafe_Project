import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS movements
                 (timestamp TEXT, direction TEXT, count INTEGER, class_name TEXT)''')
    conn.commit()
    conn.close()

def log_movement(direction, count, class_name):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''INSERT INTO movements (timestamp, direction, count, class_name)
                 VALUES (?, ?, ?, ?)''',
              (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), direction, count, class_name))
    conn.commit()
    conn.close()

def get_analytics():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''SELECT timestamp, direction, count, class_name FROM movements ORDER BY timestamp DESC''')
    rows = c.fetchall()
    conn.close()
    return rows

def clear_table():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''DELETE FROM movements''')
    conn.commit()
    conn.close()
