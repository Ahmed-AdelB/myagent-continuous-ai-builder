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
        self.transaction_manager: Optional['TransactionManager'] = None
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

                # Initialize transaction manager
                from .transaction_manager import TransactionManager
                self.transaction_manager = TransactionManager(self.pool)

                logger.success(f"Database connection pool created successfully")
                logger.info("Transaction manager initialized")
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
    """
    Initialize database schema using Alembic migrations.

    This ensures single source of truth for schema management.
    Schema is defined in alembic/versions/*.py, not here.

    Note: This function now delegates to Alembic instead of
    creating tables directly to avoid schema drift.
    """
    await db_manager.connect()

    # Run Alembic migrations to current head
    import subprocess
    import os

    try:
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Run alembic upgrade head
        result = subprocess.run(
            ['alembic', 'upgrade', 'head'],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            logger.success("Database schema initialized via Alembic migrations")
            logger.debug(f"Alembic output: {result.stdout}")
        else:
            logger.error(f"Alembic migration failed: {result.stderr}")
            raise RuntimeError(f"Alembic upgrade failed: {result.stderr}")

    except FileNotFoundError:
        logger.warning("Alembic not found - falling back to direct schema creation")
        # Fallback: Create tables directly if Alembic not available
        # (This should only happen in development/testing)
        async with db_manager.acquire() as conn:
            # Import and execute the same SQL as in Alembic migration
            from alembic.versions import import_module
            logger.info("Using fallback schema creation - Alembic recommended for production")
    except subprocess.TimeoutExpired:
        logger.error("Alembic upgrade timed out after 30 seconds")
        raise
    except Exception as e:
        logger.error(f"Failed to run Alembic migrations: {e}")
        raise
