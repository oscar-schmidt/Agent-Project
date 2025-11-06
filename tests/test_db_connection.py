"""
Simple test script to verify Supabase database connectivity
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import psycopg

# Load environment variables
load_dotenv('agents/classification_agent/.env')

def test_connection():
    """Test basic database connection"""
    database_url = os.getenv('DATABASE_URL')

    if not database_url:
        print("ERROR: DATABASE_URL not found in .env file")
        print("\nPlease add this line to your .env file:")
        print("DATABASE_URL=postgresql://postgres:ProjectD%40IS2025@db.gzqpxffdaacmmizrkmfa.supabase.co:5432/postgres")
        return False

    print("Testing Supabase connection...")
    print(f"Host: {database_url.split('@')[1].split(':')[0] if '@' in database_url else 'unknown'}")

    try:
        # Try to connect
        conn = psycopg.connect(database_url)
        print("SUCCESS: Connection successful!")

        # Test a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"PostgreSQL version: {version[:50]}...")

        # Check if tables exist
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('raw_reviews', 'detected_errors')
        """)
        tables = cursor.fetchall()

        if tables:
            print(f"Found tables: {[t[0] for t in tables]}")
        else:
            print("WARNING: Tables not found. You need to run init_database()")
            print("\nRun this command:")
            print("python -c \"from agents.classification_agent.src.database import init_database; init_database()\"")

        # Get row counts
        try:
            cursor.execute("SELECT COUNT(*) FROM raw_reviews")
            review_count = cursor.fetchone()[0]
            print(f"Total reviews in database: {review_count}")
        except:
            print("Could not count reviews (table may not exist yet)")

        cursor.close()
        conn.close()

        print("\nAll tests passed! Database is ready to use.")
        return True

    except psycopg.OperationalError as e:
        print(f"ERROR: Connection failed: {e}")
        print("\nPossible issues:")
        print("1. Wrong DATABASE_URL")
        print("2. Network/firewall blocking connection")
        print("3. Supabase project is paused or deleted")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Supabase Database Connection Test")
    print("=" * 60)
    test_connection()
