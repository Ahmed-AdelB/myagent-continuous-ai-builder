"""
Database connection management with connection pooling and retry logic.
"""

import asyncio
import asyncpg
from typing import Optional
from loguru import logger
from contextlib import asynccontextmanager

from .settings import settings


class DatabaseManager:
    """Manages PostgreSQL database connections with pooling and error handling"""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self._connection_retries = 3
        self._retry_delay = 2  # seconds

    async def connect(self):
        """Create database connection pool with retry logic"""
        for attempt in range(self._connection_retries):
            try:
                logger.info(f"Connecting to database (attempt {attempt + 1}/{self._connection_retries})...")

                self.pool = await asyncpg.create_pool(
                    dsn=settings.DATABASE_URL,
                    min_size=5,
                    max_size=settings.DATABASE_POOL_SIZE,
                    max_inactive_connection_lifetime=300,
                    command_timeout=60,
                    server_settings={
                        'application_name': 'myagent',
                        'jit': 'off'
                    }
                )

                # Test connection
                async with self.pool.acquire() as conn:
                    await conn.fetchval('SELECT 1')

                logger.success(f"Database connection pool created successfully")
                return

            except Exception as e:
                logger.error(f"Database connection failed (attempt {attempt + 1}): {e}")

                if attempt < self._connection_retries - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
                else:
                    logger.critical("Failed to connect to database after all retries")
                    raise

    async def disconnect(self):
        """Close database connection pool"""
        if self.pool:
            try:
                await self.pool.close()
                logger.info("Database connection pool closed")
            except Exception as e:
                logger.error(f"Error closing database pool: {e}")

    @asynccontextmanager
    async def acquire(self):
        """Acquire a database connection from the pool"""
        if not self.pool:
            await self.connect()

        async with self.pool.acquire() as connection:
            yield connection

    async def execute(self, query: str, *args, timeout: float = 30):
        """Execute a query with retry logic"""
        async with self.acquire() as conn:
            try:
                return await conn.execute(query, *args, timeout=timeout)
            except asyncpg.PostgresError as e:
                logger.error(f"Database query failed: {e}")
                raise

    async def fetch(self, query: str, *args, timeout: float = 30):
        """Fetch query results with retry logic"""
        async with self.acquire() as conn:
            try:
                return await conn.fetch(query, *args, timeout=timeout)
            except asyncpg.PostgresError as e:
                logger.error(f"Database query failed: {e}")
                raise

    async def fetchrow(self, query: str, *args, timeout: float = 30):
        """Fetch single row with retry logic"""
        async with self.acquire() as conn:
            try:
                return await conn.fetchrow(query, *args, timeout=timeout)
            except asyncpg.PostgresError as e:
                logger.error(f"Database query failed: {e}")
                raise

    async def fetchval(self, query: str, *args, timeout: float = 30):
        """Fetch single value with retry logic"""
        async with self.acquire() as conn:
            try:
                return await conn.fetchval(query, *args, timeout=timeout)
            except asyncpg.PostgresError as e:
                logger.error(f"Database query failed: {e}")
                raise

    async def health_check(self) -> bool:
        """Check if database connection is healthy"""
        try:
            if not self.pool:
                return False

            async with self.acquire() as conn:
                result = await conn.fetchval('SELECT 1', timeout=5)
                return result == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()


async def get_db():
    """Dependency for getting database connection"""
    if not db_manager.pool:
        await db_manager.connect()

    async with db_manager.acquire() as conn:
        yield conn


async def init_database():
    """Initialize database schema"""
    await db_manager.connect()

    # Create tables if they don't exist
    async with db_manager.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL,
                spec JSONB NOT NULL,
                state VARCHAR(50) NOT NULL DEFAULT 'initializing',
                metrics JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id VARCHAR(255) PRIMARY KEY,
                project_id INTEGER REFERENCES projects(id),
                type VARCHAR(100) NOT NULL,
                description TEXT,
                priority INTEGER,
                assigned_agent VARCHAR(100),
                status VARCHAR(50) DEFAULT 'pending',
                data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS iterations (
                id SERIAL PRIMARY KEY,
                project_id INTEGER REFERENCES projects(id),
                iteration_number INTEGER NOT NULL,
                state VARCHAR(50),
                metrics JSONB,
                tasks_completed INTEGER DEFAULT 0,
                tasks_failed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(name)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_project_id ON tasks(project_id)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_iterations_project_id ON iterations(project_id)
        """)

    logger.success("Database schema initialized successfully")
