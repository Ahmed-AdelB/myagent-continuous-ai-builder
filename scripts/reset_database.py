#!/usr/bin/env python3
"""
Database Reset Script for MyAgent System
Drops and recreates the database for clean development.
"""

import os
import sys
import asyncio
import asyncpg
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DEFAULT_DATABASE_URL = "postgresql://postgres:password@localhost:5432/myagent_db"

async def reset_database(connection_url: str) -> bool:
    """Drop and recreate the database."""
    try:
        # Parse the URL to get database name
        parts = connection_url.split('/')
        db_name = parts[-1]
        base_url = '/'.join(parts[:-1]) + '/postgres'

        # Connect to postgres database
        conn = await asyncpg.connect(base_url)

        # Terminate existing connections to the database
        await conn.execute(f"""
            SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = '{db_name}'
                AND pid <> pg_backend_pid()
        """)

        # Drop database if exists
        await conn.execute(f"DROP DATABASE IF EXISTS {db_name}")
        print(f"✓ Dropped database: {db_name}")

        # Create database
        await conn.execute(f"CREATE DATABASE {db_name}")
        print(f"✓ Created database: {db_name}")

        await conn.close()
        return True

    except Exception as e:
        print(f"✗ Error resetting database: {e}")
        return False

def main():
    """Main reset function."""
    print("⚠️  MyAgent Database Reset")
    print("=" * 50)
    print("This will DELETE ALL DATA in the database!")

    # Confirm deletion
    response = input("Are you sure? Type 'yes' to continue: ")
    if response.lower() != 'yes':
        print("Reset cancelled.")
        sys.exit(0)

    # Get database URL from environment or use default
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    print(f"Database URL: {database_url.replace('password', '***')}")

    async def run_reset():
        return await reset_database(database_url)

    # Run the reset
    success = asyncio.run(run_reset())

    if success:
        print("\n✓ Database reset completed!")
        print("Run 'python scripts/setup_database.py' to initialize schema.")
    else:
        print("\n✗ Database reset failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()