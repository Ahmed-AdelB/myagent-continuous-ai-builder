#!/bin/bash
echo "🏭 Starting MyAgent Production Environment..."

# Set production environment variables
export NODE_ENV=production
export DEV_MODE=false
export LOG_LEVEL=INFO

# Check required environment variables
if [ -z "$POSTGRES_URL" ]; then
    echo "❌ POSTGRES_URL environment variable not set"
    exit 1
fi

if [ -z "$REDIS_URL" ]; then
    echo "❌ REDIS_URL environment variable not set"
    exit 1
fi

if [ -z "$SECRET_KEY" ]; then
    echo "❌ SECRET_KEY environment variable not set"
    exit 1
fi

# Activate Python environment
echo "🐍 Activating Python environment..."
source venv/bin/activate

# Run database migrations
echo "🗄️  Running database migrations..."
python scripts/migrate_database.py

# Build frontend if exists
if [ -d "frontend" ]; then
    echo "🏗️  Building React frontend..."
    cd frontend
    npm ci --production
    npm run build
    cd ..
    echo "✅ Frontend built successfully"
else
    echo "⚠️  Frontend directory not found, skipping build"
fi

# Start production server with multiple workers
echo "🚀 Starting production server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏭 MyAgent Production Server"
echo "🌐 Listening on: 0.0.0.0:8000"
echo "⚡ Workers: 4"
echo "🔒 Security: Enabled"
echo "📊 Monitoring: Enabled"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start with production settings
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --access-log \
    --log-level info \
    --no-use-colors