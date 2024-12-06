import sqlite3

def init_db():
    with sqlite3.connect('data.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS movements (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            direction TEXT,
                            count INTEGER,
                            class INTEGER,
                            frame_idx INTEGER,
                            timestamp TEXT
                          )''')
        conn.commit()

def log_movement(direction, count, cls, frame_idx, timestamp):
    with sqlite3.connect('data.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO movements (direction, count, class, frame_idx, timestamp)
                          VALUES (?, ?, ?, ?, ?)''', (direction, count, cls, frame_idx, timestamp))
        conn.commit()

def get_analytics():
    with sqlite3.connect('data.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''SELECT timestamp, direction, count, class, frame_idx FROM movements''')
        return cursor.fetchall()

def clear_table():
    with sqlite3.connect('data.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''DELETE FROM movements''')
        conn.commit()
