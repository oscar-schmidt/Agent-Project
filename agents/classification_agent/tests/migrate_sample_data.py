#!/usr/bin/env python3
"""Migrate sample reviews from local DB to Supabase"""

import os
import psycopg
from dotenv import load_dotenv

# Load Supabase connection
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

# Local DB config
LOCAL_DB_CONFIG = {
    'host': 'localhost',
    'dbname': 'reviews',
    'user': os.getenv('DB_USER', 'reviewuser'),
    'port': '5432'
}
if os.getenv('DB_PASSWORD'):
    LOCAL_DB_CONFIG['password'] = os.getenv('DB_PASSWORD')

# Connect to local DB
print("Connecting to local database...")
local_conn = psycopg.connect(**LOCAL_DB_CONFIG)
local_cursor = local_conn.cursor()

# Fetch 10 sample reviews
local_cursor.execute("""
    SELECT review_id, review, username, email, date, reviewer_name, rating
    FROM raw_reviews
    LIMIT 10
""")
reviews = local_cursor.fetchall()
print(f"Fetched {len(reviews)} reviews from local database")

local_cursor.close()
local_conn.close()

# Connect to Supabase
print("\nConnecting to Supabase...")
supabase_url = os.getenv('DATABASE_URL')
supabase_conn = psycopg.connect(supabase_url)
supabase_cursor = supabase_conn.cursor()

# Insert into Supabase
print("Inserting reviews into Supabase...")
inserted_count = 0
for review in reviews:
    try:
        supabase_cursor.execute("""
            INSERT INTO raw_reviews (review_id, review, username, email, date, reviewer_name, rating, processed)
            VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE)
            ON CONFLICT (review_id) DO NOTHING
        """, review)
        inserted_count += 1
    except Exception as e:
        print(f"Error inserting {review[0]}: {e}")

supabase_conn.commit()
print(f"\n✅ Successfully migrated {inserted_count} reviews to Supabase")

# Verify
supabase_cursor.execute("SELECT COUNT(*) FROM raw_reviews")
total = supabase_cursor.fetchone()[0]
print(f"Total reviews in Supabase: {total}")

# Show sample
supabase_cursor.execute("SELECT review_id, reviewer_name, rating FROM raw_reviews LIMIT 5")
samples = supabase_cursor.fetchall()
print("\nSample reviews:")
for review_id, name, rating in samples:
    print(f"  - {review_id}: {name} (Rating: {rating})")

supabase_cursor.close()
supabase_conn.close()

print("\n✅ Migration complete!")
