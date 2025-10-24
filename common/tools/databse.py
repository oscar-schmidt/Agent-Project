from langchain_core.tools import BaseTool
import sqlite3


def sqlite_interact(path: str) -> None:
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS reviews (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       item_id INTEGER,
                       rating INTEGER,
                       content TEXT,
                       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                       
                      )''')
    conn.commit()
    conn.close()

class DatabaseTool(BaseTool):
    """
    Tool that can write into a sqlite database *IMPORTNAT update to be able to read or write
    """
    name: str = "DatabaseTool"
    description: str = "A tool that can interact with a database. Input should contain the thing you want to put in the databse."

    def _run(self, item_id: int| None, rating: int| None, content: str) -> str:
        path = 'db/review.db'
        sqlite_interact(path)
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                        INSERT INTO reviews (item_id, rating, content)
                        VALUES (?, ?, ?)
                        ''', (item_id, rating, content))
        except sqlite3.IntegrityError:
            return "Failed to insert data due to integrity error."
        finally:
            conn.commit()
            conn.close()
            return f"Review added successfully"

