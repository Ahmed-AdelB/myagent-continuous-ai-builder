#!/bin/bash
echo "🏭 Starting MyAgent Production Environment..."

# Build frontend
cd frontend && npm run build
cd ..

# Start with production settings
export NODE_ENV=production
export DEV_MODE=false

# Start services
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
