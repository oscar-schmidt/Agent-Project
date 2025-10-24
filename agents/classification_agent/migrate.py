#!/usr/bin/env python3
"""
Migration script to move from CSV to PostgreSQL
"""
import os
from src.database import init_database, migrate_csv_to_db
from src.config import DATA_PATH

def main():
    print("Starting database migration...")

    # Initialize database
    print(" Creating database tables...")
    init_database()

    # Migrate CSV data
    print(f" Migrating data from {DATA_PATH}...")
    migrate_csv_to_db(DATA_PATH)

    print("Migration completed successfully!")
    print("\nTo use the database instead of CSV, set:")
    print("export USE_DATABASE=true")

if __name__ == "__main__":
    main()