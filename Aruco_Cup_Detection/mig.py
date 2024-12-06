import sqlite3

def migrate_db():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()

    # Check if the column already exists
    c.execute("PRAGMA table_info(movements)")
    columns = [column[1] for column in c.fetchall()]

    if 'frame_idx' not in columns:
        c.execute('''ALTER TABLE movements ADD COLUMN frame_idx INTEGER''')

    conn.commit()
    conn.close()

if __name__ == "__main__":
    migrate_db()
    print("Migration completed successfully.")
