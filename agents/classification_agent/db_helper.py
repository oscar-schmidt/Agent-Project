#!/usr/bin/env python3
"""
Database helper script for common operations
"""
import sys
from src.database import get_connection, get_processing_stats

def show_stats():
    """Show processing statistics"""
    stats = get_processing_stats()
    print(" Database Statistics:")
    print(f"  Total reviews: {stats['total']}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Unprocessed: {stats['unprocessed']}")

def show_recent_reviews(limit=5):
    """Show recent reviews"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT review_id, reviewer_name, rating, processed, created_at
        FROM raw_reviews
        ORDER BY created_at DESC
        LIMIT %s
    """, (limit,))

    rows = cursor.fetchall()
    print(f"\n Recent {limit} reviews:")
    for row in rows:
        status = "" if row[3] else ""
        print(f"  {status} {row[0]} | {row[1]} | Rating: {row[2]} | {row[4]}")

    cursor.close()
    conn.close()

def mark_review_unprocessed(review_id):
    """Mark a review as unprocessed for testing"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("UPDATE raw_reviews SET processed = FALSE WHERE review_id = %s", (review_id,))
    affected = cursor.rowcount

    conn.commit()
    cursor.close()
    conn.close()

    if affected:
        print(f" Marked {review_id} as unprocessed")
    else:
        print(f" Review {review_id} not found")

def search_reviews(search_term):
    """Search reviews by content"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT review_id, reviewer_name, review, processed
        FROM raw_reviews
        WHERE review ILIKE %s
        ORDER BY created_at DESC
        LIMIT 10
    """, (f"%{search_term}%",))

    rows = cursor.fetchall()
    print(f"\n Found {len(rows)} reviews containing '{search_term}':")
    for row in rows:
        status = "" if row[3] else ""
        print(f"  {status} {row[0]} | {row[1]}")
        print(f"    {row[2][:100]}...")

    cursor.close()
    conn.close()

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python db_helper.py stats                    # Show statistics")
        print("  python db_helper.py recent [N]               # Show recent N reviews")
        print("  python db_helper.py unprocess REV-XXXX       # Mark review as unprocessed")
        print("  python db_helper.py search 'crash'          # Search for reviews")
        return

    command = sys.argv[1]

    if command == "stats":
        show_stats()
    elif command == "recent":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        show_recent_reviews(limit)
    elif command == "unprocess":
        if len(sys.argv) < 3:
            print("Please provide review ID: python db_helper.py unprocess REV-0001")
            return
        mark_review_unprocessed(sys.argv[2])
    elif command == "search":
        if len(sys.argv) < 3:
            print("Please provide search term: python db_helper.py search 'crash'")
            return
        search_reviews(sys.argv[2])
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()