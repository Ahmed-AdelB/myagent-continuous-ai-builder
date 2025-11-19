# Database Schema Management

## Overview

This project uses **Alembic** for database schema management, ensuring a single source of truth for all database changes.

## Single Source of Truth

**IMPORTANT**: Database schema is defined ONLY in Alembic migrations (`alembic/versions/*.py`).

Do **NOT** create tables or modify schema in:
- `config/database.py` (connection management only)
- API endpoints
- Agent code
- Tests (use temporary tables or fixtures)

## Architecture

```
alembic/
├── README.md           # This file
├── env.py             # Alembic environment configuration
└── versions/          # Migration files (single source of truth)
    └── 0001_init.py   # Initial schema

config/
└── database.py        # Connection pool + delegates to Alembic
```

## How It Works

### Initialization

When `init_database()` is called:

1. **Connects** to PostgreSQL database
2. **Runs** `alembic upgrade head` to apply all migrations
3. **Logs** success or detailed error messages
4. **Falls back** to direct creation only if Alembic unavailable (dev/test only)

### Migration Flow

```
Code Change → Create Migration → Review → Apply → Deploy
     ↓              ↓              ↓        ↓         ↓
Need new     alembic revision   Git PR   alembic   Production
  table        -m "message"     review   upgrade    servers
```

## Commands

### Create New Migration

```bash
# Auto-generate migration from ORM changes
alembic revision --autogenerate -m "Add new table"

# Create empty migration (for data migrations)
alembic revision -m "Migrate legacy data"
```

### Apply Migrations

```bash
# Upgrade to latest
alembic upgrade head

# Upgrade one version
alembic upgrade +1

# Downgrade one version
alembic downgrade -1

# Go to specific version
alembic upgrade <revision_id>
```

### Check Status

```bash
# Show current version
alembic current

# Show migration history
alembic history

# Show pending migrations
alembic heads
```

## Current Schema

### Tables

**projects**
- `id` (SERIAL PRIMARY KEY)
- `name` (VARCHAR(255) UNIQUE NOT NULL)
- `spec` (JSONB NOT NULL)
- `state` (VARCHAR(50) NOT NULL DEFAULT 'initializing')
- `metrics` (JSONB)
- `created_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
- `updated_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)

**tasks**
- `id` (VARCHAR(255) PRIMARY KEY)
- `project_id` (INTEGER REFERENCES projects(id))
- `type` (VARCHAR(100) NOT NULL)
- `description` (TEXT)
- `priority` (INTEGER)
- `assigned_agent` (VARCHAR(100))
- `status` (VARCHAR(50) DEFAULT 'pending')
- `data` (JSONB)
- `created_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
- `started_at` (TIMESTAMP)
- `completed_at` (TIMESTAMP)

**iterations**
- `id` (SERIAL PRIMARY KEY)
- `project_id` (INTEGER REFERENCES projects(id))
- `iteration_number` (INTEGER NOT NULL)
- `state` (VARCHAR(50))
- `metrics` (JSONB)
- `tasks_completed` (INTEGER DEFAULT 0)
- `tasks_failed` (INTEGER DEFAULT 0)
- `created_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)

### Indexes

- `idx_projects_name` ON `projects(name)`
- `idx_tasks_project_id` ON `tasks(project_id)`
- `idx_tasks_status` ON `tasks(status)`
- `idx_iterations_project_id` ON `iterations(project_id)`

## Best Practices

### 1. Never Direct SQL

```python
# ❌ BAD - Creates schema drift
await conn.execute("CREATE TABLE users (...)")

# ✅ GOOD - Use Alembic migration
# alembic revision -m "Add users table"
```

### 2. Always Review Migrations

```bash
# Before committing, check what Alembic will do
cat alembic/versions/<new_migration>.py

# Test migration on dev database
alembic upgrade head

# Test rollback works
alembic downgrade -1
alembic upgrade +1
```

### 3. Data Migrations

For data transformations, use separate migrations:

```python
# alembic/versions/0002_migrate_old_data.py
def upgrade():
    # First: Structural change
    op.add_column('users', sa.Column('status', sa.String(50)))

    # Second: Data migration
    op.execute("""
        UPDATE users SET status = 'active' WHERE last_login > NOW() - INTERVAL '30 days'
    """)

    # Third: Apply constraints
    op.alter_column('users', 'status', nullable=False)
```

### 4. Testing Migrations

```python
# tests/test_migrations/test_001_init.py
async def test_initial_migration(test_db):
    # Apply migration
    await run_migration('0001_init')

    # Verify tables exist
    tables = await test_db.fetch("SELECT tablename FROM pg_tables WHERE schemaname='public'")
    table_names = [t['tablename'] for t in tables]

    assert 'projects' in table_names
    assert 'tasks' in table_names
    assert 'iterations' in table_names
```

## Troubleshooting

### Migration Already Applied

```bash
# Error: Target database is not up to date
alembic stamp head

# Or manually set version
alembic stamp <revision_id>
```

### Schema Drift Detected

```bash
# Compare runtime schema to migrations
alembic check

# Auto-generate migration to sync
alembic revision --autogenerate -m "Fix schema drift"
```

### Database Reset (Development Only)

```bash
# WARNING: Destroys all data!
dropdb myagent_db
createdb myagent_db
alembic upgrade head
```

## Production Deployment

### Pre-Deployment Checklist

- [ ] Migration tested on dev database
- [ ] Migration tested on staging database
- [ ] Rollback tested (downgrade → upgrade)
- [ ] Data backed up
- [ ] Migration reviewed by team
- [ ] Performance impact assessed

### Deployment Process

```bash
# 1. Backup database
pg_dump myagent_db > backup_$(date +%Y%m%d).sql

# 2. Apply migration
alembic upgrade head

# 3. Verify success
alembic current
psql -c "SELECT COUNT(*) FROM projects"

# 4. If failed, rollback
alembic downgrade -1
```

## Configuration

### Environment Variables

```bash
# Override database URL
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"

# Then run migrations
alembic upgrade head
```

### alembic.ini

Default connection: `postgresql://myagent:myagent_password@localhost:5432/myagent_db`

Override with `DATABASE_URL` environment variable (recommended for production).

## References

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- Project migrations: `alembic/versions/`
- Configuration: `alembic.ini` and `alembic/env.py`

---

**Last Updated**: 2025-11-19
**Current Schema Version**: 0001_init
**Migration Count**: 1
