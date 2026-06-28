import sqlite3

def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            prediction TEXT,
            confidence REAL
        )
    """)

    conn.commit()
    conn.close()


def insert_log(filename, prediction, confidence):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO logs (filename, prediction, confidence)
        VALUES (?, ?, ?)
    """, (filename, prediction, confidence))

    conn.commit()
    conn.close()