#!/usr/bin/env python3
"""
Database Setup Script for MyAgent System
Initializes PostgreSQL database with all required tables and indexes.
"""

import os
import sys
import asyncio
import asyncpg
from pathlib import Path
from typing import Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DEFAULT_DATABASE_URL = "postgresql://postgres:password@localhost:5432/myagent_db"

async def create_database_if_not_exists(connection_url: str) -> bool:
    """Create the database if it doesn't exist."""
    try:
        # Parse the URL to get database name
        parts = connection_url.split('/')
        db_name = parts[-1]
        base_url = '/'.join(parts[:-1]) + '/postgres'

        # Connect to postgres database to create our database
        conn = await asyncpg.connect(base_url)

        # Check if database exists
        result = await conn.fetch("SELECT 1 FROM pg_database WHERE datname = $1", db_name)

        if not result:
            await conn.execute(f"CREATE DATABASE {db_name}")
            print(f"✓ Created database: {db_name}")
        else:
            print(f"✓ Database {db_name} already exists")

        await conn.close()
        return True

    except Exception as e:
        print(f"✗ Error creating database: {e}")
        return False

async def setup_database(connection_url: str) -> bool:
    """Set up the MyAgent database with all required tables."""
    try:
        # First ensure database exists
        if not await create_database_if_not_exists(connection_url):
            return False

        # Connect to the target database
        conn = await asyncpg.connect(connection_url)

        # Read and execute the SQL schema
        sql_file = project_root / "scripts" / "init_database.sql"

        if not sql_file.exists():
            print(f"✗ SQL file not found: {sql_file}")
            return False

        with open(sql_file, 'r') as f:
            sql_content = f.read()

        # Execute the schema
        await conn.execute(sql_content)
        print("✓ Database schema created successfully")

        # Verify tables were created
        tables = await conn.fetch("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)

        table_names = [row['table_name'] for row in tables]
        print(f"✓ Created {len(table_names)} tables: {', '.join(table_names)}")

        await conn.close()
        return True

    except Exception as e:
        print(f"✗ Error setting up database: {e}")
        return False

async def test_database_connection(connection_url: str) -> bool:
    """Test the database connection and basic functionality."""
    try:
        conn = await asyncpg.connect(connection_url)

        # Test basic query
        result = await conn.fetchval("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
        print(f"✓ Database connection successful. Found {result} tables.")

        await conn.close()
        return True

    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 MyAgent Database Setup")
    print("=" * 50)

    # Get database URL from environment or use default
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    print(f"Database URL: {database_url.replace('password', '***')}")

    async def run_setup():
        # Setup database
        if await setup_database(database_url):
            print("\n✓ Database setup completed successfully!")

            # Test connection
            if await test_database_connection(database_url):
                print("✓ Database is ready for use!")
                return True

        print("\n✗ Database setup failed!")
        return False

    # Run the setup
    success = asyncio.run(run_setup())

    if success:
        print("\n🎉 MyAgent database is ready!")
        print("You can now start the application.")
        sys.exit(0)
    else:
        print("\n💥 Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()