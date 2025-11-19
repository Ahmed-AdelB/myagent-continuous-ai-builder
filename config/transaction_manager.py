"""
Transaction Manager for Database Operations

Provides transaction contexts for atomic database operations,
ensuring consistency during concurrent access and error scenarios.

Part of P3 Item #18: Transaction Boundaries
"""

import asyncio
from typing import Optional, Any, Callable
from contextlib import asynccontextmanager
from loguru import logger
import asyncpg


class TransactionManager:
    """
    Manages database transactions with proper isolation and rollback

    Provides:
    - Atomic transaction contexts
    - Automatic rollback on errors
    - Nested transaction support (savepoints)
    - Isolation level control
    - Deadlock detection and retry
    """

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self.default_isolation = 'read_committed'
        self.max_retries = 3
        self.retry_delay = 0.5  # seconds

    @asynccontextmanager
    async def transaction(
        self,
        isolation: Optional[str] = None,
        readonly: bool = False,
        deferrable: bool = False
    ):
        """
        Context manager for database transactions

        Args:
            isolation: Isolation level (read_committed, repeatable_read, serializable)
            readonly: If True, transaction is read-only
            deferrable: If True, transaction can be deferred

        Usage:
            async with transaction_manager.transaction():
                await conn.execute("INSERT INTO ...")
                await conn.execute("UPDATE ...")
                # Auto-commit on success, rollback on exception
        """
        isolation = isolation or self.default_isolation

        async with self.pool.acquire() as conn:
            tx = conn.transaction(
                isolation=isolation,
                readonly=readonly,
                deferrable=deferrable
            )

            try:
                await tx.start()
                logger.debug(f"Transaction started (isolation={isolation})")

                yield conn

                await tx.commit()
                logger.debug("Transaction committed successfully")

            except asyncpg.SerializationError as e:
                await tx.rollback()
                logger.warning(f"Serialization error, transaction rolled back: {e}")
                raise

            except asyncpg.DeadlockDetectedError as e:
                await tx.rollback()
                logger.error(f"Deadlock detected, transaction rolled back: {e}")
                raise

            except Exception as e:
                await tx.rollback()
                logger.error(f"Transaction failed, rolled back: {e}")
                raise

    @asynccontextmanager
    async def savepoint(self, conn: asyncpg.Connection, name: str = "sp"):
        """
        Create a savepoint for nested transactions

        Args:
            conn: Database connection
            name: Savepoint name

        Usage:
            async with transaction_manager.transaction() as conn:
                await conn.execute("INSERT ...")

                async with transaction_manager.savepoint(conn, "inner"):
                    await conn.execute("UPDATE ...")
                    # Can rollback to savepoint without affecting outer transaction
        """
        try:
            await conn.execute(f"SAVEPOINT {name}")
            logger.debug(f"Savepoint created: {name}")

            yield conn

            await conn.execute(f"RELEASE SAVEPOINT {name}")
            logger.debug(f"Savepoint released: {name}")

        except Exception as e:
            await conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
            logger.warning(f"Rolled back to savepoint {name}: {e}")
            raise

    async def execute_with_retry(
        self,
        operation: Callable,
        *args,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Execute operation with automatic retry on serialization failures

        Args:
            operation: Async function to execute
            max_retries: Max retry attempts (default: self.max_retries)
            *args, **kwargs: Arguments for operation

        Returns:
            Result of operation

        Raises:
            Exception if all retries exhausted
        """
        max_retries = max_retries or self.max_retries
        last_error = None

        for attempt in range(max_retries):
            try:
                async with self.transaction() as conn:
                    result = await operation(conn, *args, **kwargs)
                    return result

            except asyncpg.SerializationError as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Serialization error (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max retries ({max_retries}) exhausted")

            except asyncpg.DeadlockDetectedError as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Deadlock detected (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max retries ({max_retries}) exhausted")

            except Exception as e:
                # Non-retriable error
                logger.error(f"Operation failed with non-retriable error: {e}")
                raise

        # All retries exhausted
        raise last_error

    async def batch_insert(
        self,
        table: str,
        columns: list,
        values: list,
        batch_size: int = 1000
    ):
        """
        Insert large batches of data in transactions

        Args:
            table: Table name
            columns: List of column names
            values: List of tuples with values
            batch_size: Number of rows per transaction

        Returns:
            Total number of rows inserted
        """
        total_inserted = 0

        for i in range(0, len(values), batch_size):
            batch = values[i:i + batch_size]

            async with self.transaction() as conn:
                # Build query
                placeholders = ', '.join([
                    f"({', '.join([f'${j*len(columns) + k + 1}' for k in range(len(columns))])})"
                    for j in range(len(batch))
                ])

                query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES {placeholders}"

                # Flatten batch
                flat_values = [val for row in batch for val in row]

                await conn.execute(query, *flat_values)
                total_inserted += len(batch)

                logger.debug(f"Inserted batch: {len(batch)} rows into {table}")

        logger.info(f"Batch insert complete: {total_inserted} total rows into {table}")
        return total_inserted

    async def atomic_update(
        self,
        table: str,
        updates: dict,
        where_clause: str,
        where_values: list
    ) -> int:
        """
        Perform atomic update with WHERE clause

        Args:
            table: Table name
            updates: Dict of column: value pairs to update
            where_clause: WHERE clause (e.g., "id = $1")
            where_values: Values for WHERE clause

        Returns:
            Number of rows updated
        """
        async with self.transaction() as conn:
            # Build SET clause
            set_parts = []
            set_values = []
            param_idx = 1

            for col, val in updates.items():
                set_parts.append(f"{col} = ${param_idx}")
                set_values.append(val)
                param_idx += 1

            set_clause = ', '.join(set_parts)

            # Adjust WHERE clause parameter indices
            adjusted_where = where_clause
            for i in range(len(where_values)):
                adjusted_where = adjusted_where.replace(f'${i + 1}', f'${param_idx + i}')

            # Build full query
            query = f"UPDATE {table} SET {set_clause} WHERE {adjusted_where}"
            all_values = set_values + where_values

            result = await conn.execute(query, *all_values)

            # Extract row count from result
            rows_updated = int(result.split()[-1])
            logger.debug(f"Updated {rows_updated} rows in {table}")

            return rows_updated

    async def atomic_delete(
        self,
        table: str,
        where_clause: str,
        where_values: list,
        require_where: bool = True
    ) -> int:
        """
        Perform atomic delete with WHERE clause

        Args:
            table: Table name
            where_clause: WHERE clause
            where_values: Values for WHERE clause
            require_where: If True, requires WHERE clause (safety)

        Returns:
            Number of rows deleted
        """
        if require_where and not where_clause:
            raise ValueError("WHERE clause required for safety (use require_where=False to override)")

        async with self.transaction() as conn:
            if where_clause:
                query = f"DELETE FROM {table} WHERE {where_clause}"
                result = await conn.execute(query, *where_values)
            else:
                query = f"DELETE FROM {table}"
                result = await conn.execute(query)

            rows_deleted = int(result.split()[-1])
            logger.debug(f"Deleted {rows_deleted} rows from {table}")

            return rows_deleted

    async def get_stats(self) -> dict:
        """Get transaction statistics"""
        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow("""
                SELECT
                    xact_commit,
                    xact_rollback,
                    deadlocks,
                    blk_read_time,
                    blk_write_time
                FROM pg_stat_database
                WHERE datname = current_database()
            """)

            return dict(stats) if stats else {}
