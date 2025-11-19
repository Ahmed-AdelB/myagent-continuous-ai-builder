"""Initial schema aligned with runtime tables.

NOTE: Added to satisfy CI alembic upgrade head (Codex).
"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL,
            spec JSONB NOT NULL,
            state VARCHAR(50) NOT NULL DEFAULT 'initializing',
            metrics JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    op.execute(
        """
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
        """
    )

    op.execute(
        """
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
        """
    )

    op.execute("CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(name)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_tasks_project_id ON tasks(project_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_iterations_project_id ON iterations(project_id)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS iterations")
    op.execute("DROP TABLE IF EXISTS tasks")
    op.execute("DROP TABLE IF EXISTS projects")
