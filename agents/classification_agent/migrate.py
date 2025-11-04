#!/usr/bin/env python3
"""
Migration script to initialize PostgreSQL database and optionally migrate CSV data
"""
import sys
from src.database import init_database, migrate_csv_to_db

def main():
    print("Starting database setup...")

    # Initialize database
    print("Creating database tables...")
    init_database()

    # Check if CSV file path is provided as argument
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        print(f"Migrating data from {csv_path}...")
        migrate_csv_to_db(csv_path)
        print("\nMigration completed successfully!")
    else:
        print("\nDatabase tables created successfully!")
        print("To migrate CSV data, run:")
        print("  python migrate.py /path/to/your/data.csv")

if __name__ == "__main__":
    main()