-- Database initialization script for MyAgent
-- PostgreSQL database setup

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS myagent_db;

\c myagent_db;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Create projects table
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    quality_score DECIMAL(5,2) DEFAULT 0.00,
    test_coverage DECIMAL(5,2) DEFAULT 0.00,
    critical_bugs INTEGER DEFAULT 0,
    performance_score DECIMAL(5,2) DEFAULT 0.00,
    security_score DECIMAL(5,2) DEFAULT 0.00
);

-- Create iterations table
CREATE TABLE IF NOT EXISTS iterations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    iteration_number INTEGER NOT NULL,
    status VARCHAR(50) DEFAULT 'running',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds INTEGER,
    objectives JSONB DEFAULT '[]',
    results JSONB DEFAULT '{}',
    quality_metrics JSONB DEFAULT '{}',
    files_modified INTEGER DEFAULT 0,
    lines_added INTEGER DEFAULT 0,
    lines_removed INTEGER DEFAULT 0,
    tests_added INTEGER DEFAULT 0
);

-- Create agents table
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'idle',
    capabilities TEXT[],
    current_task VARCHAR(500),
    tasks_completed INTEGER DEFAULT 0,
    success_rate DECIMAL(5,2) DEFAULT 100.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Create tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    iteration_id UUID REFERENCES iterations(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
    type VARCHAR(100) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    result JSONB DEFAULT '{}',
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3
);

-- Create errors table
CREATE TABLE IF NOT EXISTS errors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    iteration_id UUID REFERENCES iterations(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    type VARCHAR(100) NOT NULL,
    severity VARCHAR(50) DEFAULT 'medium',
    message TEXT NOT NULL,
    stack_trace TEXT,
    context JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'unresolved',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    resolution TEXT,
    similar_error_ids UUID[],
    occurrence_count INTEGER DEFAULT 1
);

-- Create patterns table
CREATE TABLE IF NOT EXISTS patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type VARCHAR(100) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    context JSONB NOT NULL,
    solution JSONB NOT NULL,
    success_rate DECIMAL(5,2) DEFAULT 50.00,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP,
    tags TEXT[]
);

-- Create metrics table
CREATE TABLE IF NOT EXISTS metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    iteration_id UUID REFERENCES iterations(id) ON DELETE CASCADE,
    metric_type VARCHAR(100) NOT NULL,
    value DECIMAL(10,4) NOT NULL,
    unit VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Create audit_log table
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100),
    entity_id UUID,
    changes JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create sessions table for authentication
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(500) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT
);

-- Create API keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    permissions JSONB DEFAULT '[]',
    rate_limit INTEGER DEFAULT 1000,
    expires_at TIMESTAMP,
    last_used TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Create indexes for performance
CREATE INDEX idx_projects_user_id ON projects(user_id);
CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_iterations_project_id ON iterations(project_id);
CREATE INDEX idx_iterations_status ON iterations(status);
CREATE INDEX idx_tasks_project_id ON tasks(project_id);
CREATE INDEX idx_tasks_iteration_id ON tasks(iteration_id);
CREATE INDEX idx_tasks_agent_id ON tasks(agent_id);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_errors_project_id ON errors(project_id);
CREATE INDEX idx_errors_status ON errors(status);
CREATE INDEX idx_errors_severity ON errors(severity);
CREATE INDEX idx_metrics_project_id ON metrics(project_id);
CREATE INDEX idx_metrics_timestamp ON metrics(timestamp);
CREATE INDEX idx_patterns_type ON patterns(type);
CREATE INDEX idx_patterns_success_rate ON patterns(success_rate DESC);
CREATE INDEX idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX idx_audit_log_timestamp ON audit_log(timestamp DESC);
CREATE INDEX idx_sessions_token ON sessions(token);
CREATE INDEX idx_sessions_expires_at ON sessions(expires_at);
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);

-- Create triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function to clean expired sessions
CREATE OR REPLACE FUNCTION clean_expired_sessions()
RETURNS void AS $$
BEGIN
    DELETE FROM sessions WHERE expires_at < CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Create function to calculate project quality score
CREATE OR REPLACE FUNCTION calculate_project_quality(project_uuid UUID)
RETURNS DECIMAL AS $$
DECLARE
    quality DECIMAL(5,2);
BEGIN
    SELECT (
        COALESCE(test_coverage, 0) * 0.25 +
        CASE WHEN critical_bugs = 0 THEN 25 ELSE 0 END +
        COALESCE(performance_score, 0) * 0.25 +
        COALESCE(security_score, 0) * 0.25
    ) INTO quality
    FROM projects
    WHERE id = project_uuid;

    RETURN quality;
END;
$$ LANGUAGE plpgsql;

-- Insert default admin user (password: admin123 - should be changed immediately)
INSERT INTO users (username, email, password_hash, role)
VALUES ('admin', 'admin@myagent.local', crypt('admin123', gen_salt('bf')), 'admin')
ON CONFLICT (username) DO NOTHING;

-- Insert default agents
INSERT INTO agents (name, type, capabilities) VALUES
('Coder Agent', 'coder', ARRAY['code_generation', 'refactoring', 'optimization']),
('Tester Agent', 'tester', ARRAY['test_generation', 'test_execution', 'coverage_analysis']),
('Debugger Agent', 'debugger', ARRAY['error_analysis', 'debugging', 'fix_suggestion']),
('Architect Agent', 'architect', ARRAY['design_review', 'pattern_suggestion', 'scalability_analysis']),
('Analyzer Agent', 'analyzer', ARRAY['metric_monitoring', 'trend_analysis', 'performance_tracking']),
('UI Refiner Agent', 'ui_refiner', ARRAY['ui_improvement', 'ux_optimization', 'accessibility_check'])
ON CONFLICT DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE myagent_db TO myagent_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO myagent_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO myagent_user;