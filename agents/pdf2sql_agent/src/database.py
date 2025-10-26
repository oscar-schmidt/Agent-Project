import os
import psycopg
from typing import List, Optional
from agents.pdf2sql_agent.src.utils import RawReview

# Database configuration
import getpass
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'dbname': os.getenv('DB_NAME', 'reviews'),
    'user': os.getenv('DB_USER', getpass.getuser()), 
    'port': os.getenv('DB_PORT', '5432')
}


if os.getenv('DB_PASSWORD'):
    DB_CONFIG['password'] = os.getenv('DB_PASSWORD')

def get_connection():
    """Get database connection"""
    return psycopg.connect(**DB_CONFIG)

def init_database():
    """Initialize database with required tables"""
    conn = get_connection()
    cursor = conn.cursor()

    #reviews table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_reviews (
            review_id VARCHAR(20) PRIMARY KEY,
            review TEXT NOT NULL,
            username VARCHAR(100),
            email VARCHAR(255) NOT NULL,
            date TIMESTAMP,
            reviewer_name VARCHAR(100),
            rating INTEGER CHECK (rating >= 1 AND rating <= 5),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed BOOLEAN DEFAULT FALSE
        );
    """)

    #detected_errors table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detected_errors (
            id SERIAL PRIMARY KEY,
            review_id VARCHAR(20) REFERENCES raw_reviews(review_id) ON DELETE CASCADE,
            error_summary TEXT NOT NULL,
            error_type VARCHAR(50),
            criticality VARCHAR(20),
            rationale TEXT,
            error_hash VARCHAR(64) UNIQUE,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notion_synced BOOLEAN DEFAULT FALSE
        );
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("Database initialized successfully")


#ONLY AT INIT
def migrate_csv_to_db(csv_path: str):
    """Migrate CSV data to PostgreSQL"""
    from src.nodes.load_reviews import load_reviews

   
    reviews = load_reviews(csv_path)

    conn = get_connection()
    cursor = conn.cursor()

    for review in reviews:
        cursor.execute("""
            INSERT INTO raw_reviews (review_id, review, username, email, date, reviewer_name, rating)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (review_id) DO NOTHING
        """, (
            review.review_id,
            review.review,
            review.username,
            review.email,
            review.date,
            review.reviewer_name,
            review.rating
        ))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Migrated {len(reviews)} reviews to database")

def load_unprocessed_reviews(batch_size: Optional[int] = None) -> List[RawReview]:
    """load unprocessed reviews from PostgreSQL DB"""
    conn = get_connection()
    cursor = conn.cursor()

    #default to reasonable batch size
    if batch_size is None:
        batch_size = int(os.getenv('BATCH_SIZE', '50'))

    query = """
        SELECT review_id, review, username, email, date, reviewer_name, rating
        FROM raw_reviews
        WHERE processed = FALSE
        ORDER BY created_at ASC
        LIMIT %s
    """

    cursor.execute(query, (batch_size,))
    rows = cursor.fetchall()

    reviews = []
    for row in rows:
        #DT to stirng
        date_str = row[4]
        if hasattr(date_str, 'strftime'):
            date_str = date_str.strftime('%Y-%m-%d %H:%M:%S')

        reviews.append(RawReview(
            review_id=row[0],  
            review=row[1],     
            username=row[2],   
            email=row[3],      
            date=str(date_str),
            reviewer_name=row[5],
            rating=row[6]      
        ))

    cursor.close()
    conn.close()

    if reviews:
        print(f" Loaded {len(reviews)} unprocessed reviews")
    else:
        print(" No unprocessed reviews found")

    return reviews

def load_reviews_from_db(limit: Optional[int] = None) -> List[RawReview]:
    """Legacy function - redirects to load_unprocessed_reviews"""
    return load_unprocessed_reviews(limit)

def mark_review_processed(review_id: str):
    """Mark a single review as processed"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("UPDATE raw_reviews SET processed = TRUE WHERE review_id = %s", (review_id,))

    conn.commit()
    cursor.close()
    conn.close()

def mark_reviews_processed(review_ids: List[str]):
    """Mark multiple reviews as processed in a single transaction"""
    if not review_ids:
        return

    conn = get_connection()
    cursor = conn.cursor()

    
    placeholders = ','.join(['%s'] * len(review_ids))
    query = f"UPDATE raw_reviews SET processed = TRUE WHERE review_id IN ({placeholders})"

    cursor.execute(query, review_ids)
    affected_rows = cursor.rowcount

    conn.commit()
    cursor.close()
    conn.close()

    print(f" Marked {affected_rows} reviews as processed")

def get_processing_stats():
    """Get statistics about processed vs unprocessed reviews"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN processed = TRUE THEN 1 END) as processed,
            COUNT(CASE WHEN processed = FALSE THEN 1 END) as unprocessed
        FROM raw_reviews
    """)

    row = cursor.fetchone()
    cursor.close()
    conn.close()

    return {
        'total': row[0],
        'processed': row[1],
        'unprocessed': row[2]
    }

def get_next_review_id():
    """Generate next review ID in sequence (REV-XXXX)"""
    conn = get_connection()
    cursor = conn.cursor()

    #find the highest existing review ID
    cursor.execute("""
        SELECT review_id FROM raw_reviews
        WHERE review_id ~ '^REV-[0-9]+$'
        ORDER BY CAST(SUBSTRING(review_id FROM 5) AS INTEGER) DESC
        LIMIT 1
    """)

    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row:
        #extract number and increment
        last_num = int(row[0].split('-')[1])
        next_num = last_num + 1
    else:
        #start from 1 if no reviews exist
        next_num = 1

    return f"REV-{next_num:04d}"

def insert_new_review(review_text: str, username: str, email: str,
                     reviewer_name: str, rating: int):
    """Insert a new review with auto-generated ID"""
    review_id = get_next_review_id()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO raw_reviews
        (review_id, review, username, email, date, reviewer_name, rating, processed)
        VALUES (%s, %s, %s, %s, NOW(), %s, %s, FALSE)
    """, (review_id, review_text, username, email, reviewer_name, rating))

    conn.commit()
    cursor.close()
    conn.close()

    print(f" Added new review {review_id} by {reviewer_name}")
    return review_id