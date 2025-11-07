#!/bin/bash
# Auto-sync script for 22_MyAgent
# Automatically commits and pushes all changes to GitHub

cd /home/aadel/projects/22_MyAgent

# Check if there are any changes
if [ -n "$(git status --porcelain)" ]; then
    echo "📝 Changes detected, syncing to GitHub..."
    
    # Add all changes
    git add .
    
    # Create commit with timestamp (using date command compatible with Linux)
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    git commit -m "Auto-sync: $TIMESTAMP

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
    
    # Push to GitHub
    git push origin master:main
    
    echo "✅ Synced to GitHub: https://github.com/Ahmed-AdelB/myagent-continuous-ai-builder"
else
    echo "📋 No changes to sync"
fi
