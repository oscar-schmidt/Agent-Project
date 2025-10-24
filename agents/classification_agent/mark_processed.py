#!/usr/bin/env python3
"""
Mark all existing reviews as processed to start fresh
"""
from src.database import get_connection

def mark_all_processed():
    print("Marking all existing reviews as processed...")

    conn = get_connection()
    cursor = conn.cursor()

    # Update all reviews to processed = TRUE
    cursor.execute("UPDATE raw_reviews SET processed = TRUE")
    affected_rows = cursor.rowcount

    conn.commit()
    cursor.close()
    conn.close()

    print(f" Marked {affected_rows} reviews as processed")
    print("Now only new reviews will be processed in future runs")

if __name__ == "__main__":
    mark_all_processed()