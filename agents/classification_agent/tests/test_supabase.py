#!/usr/bin/env python3
"""Test Supabase connection and initialize tables"""

import os
from dotenv import load_dotenv

# Load .env file from classification_agent directory
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

print(f"DATABASE_URL loaded: {'Yes' if os.getenv('DATABASE_URL') else 'No'}")

from agents.classification_agent.src.database import get_connection, init_database

# Test connection
conn = get_connection()
cursor = conn.cursor()

cursor.execute('SELECT current_database(), version()')
db_name, version = cursor.fetchone()
print(f"\nConnected to: {db_name}")
print(f"PostgreSQL version: {version.split(',')[0]}")

# Check if tables exist
cursor.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name IN ('raw_reviews', 'detected_errors')
    ORDER BY table_name
""")
existing_tables = [row[0] for row in cursor.fetchall()]

if len(existing_tables) == 2:
    print(f"\n✅ Tables already exist: {existing_tables}")

    # Show row counts
    cursor.execute('SELECT COUNT(*) FROM raw_reviews')
    review_count = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM detected_errors')
    error_count = cursor.fetchone()[0]

    print(f"   - raw_reviews: {review_count} rows")
    print(f"   - detected_errors: {error_count} rows")
else:
    print(f"\n⚠️  Tables missing: {set(['raw_reviews', 'detected_errors']) - set(existing_tables)}")
    print("Initializing database...")
    cursor.close()
    conn.close()
    init_database()
    print("✅ Database initialized!")

cursor.close()
conn.close()
