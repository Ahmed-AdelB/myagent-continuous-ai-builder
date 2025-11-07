#!/usr/bin/env python3
"""
GitHub Auto-Sync Setup Script for 22_MyAgent
This script sets up automatic GitHub synchronization for every change.
"""

import os
import subprocess
import sys
from pathlib import Path
import datetime

def run_command(cmd, description=""):
    """Run shell command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed: {e.stderr}")
        return None

def setup_github_autosync():
    """Setup complete GitHub auto-sync system"""
    print("\n🚀 Setting up GitHub Auto-Sync for 22_MyAgent...")
    
    # 1. Create auto-sync script with fixed date format
    autosync_script = '''#!/bin/bash
# Auto-sync script for 22_MyAgent
# Automatically commits and pushes all changes to GitHub

cd /home/aadel/projects/22_MyAgent

# Check if there are any changes
if [ -n "" ]; then
    echo "📝 Changes detected, syncing to GitHub..."
    
    # Add all changes
    git add .
    
    # Create commit with timestamp (using date command compatible with Linux)
    TIMESTAMP=2025-11-07 21:15:47
    git commit -m "Auto-sync: $TIMESTAMP

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
    
    # Push to GitHub
    git push origin master:main
    
    echo "✅ Synced to GitHub: https://github.com/Ahmed-AdelB/myagent-continuous-ai-builder"
else
    echo "📋 No changes to sync"
fi
'''

    with open('auto-sync.sh', 'w') as f:
        f.write(autosync_script)
    
    run_command('chmod +x auto-sync.sh', 'Made auto-sync script executable')
    
    # 2. Create manual sync command
    manual_sync = '''#!/bin/bash
# Manual sync function for immediate use
function myagent_sync() {
    cd /home/aadel/projects/22_MyAgent
    ./auto-sync.sh
}

# Add to shell profile if not already present
if ! grep -q "myagent_sync" ~/.bashrc; then
    echo "# MyAgent sync function" >> ~/.bashrc
    echo "function myagent_sync() { cd /home/aadel/projects/22_MyAgent && ./auto-sync.sh; }" >> ~/.bashrc
    echo "alias sync='myagent_sync'" >> ~/.bashrc
    echo "✅ Added sync command to ~/.bashrc"
else
    echo "📋 Sync command already exists in ~/.bashrc"
fi
'''

    with open('setup-sync-command.sh', 'w') as f:
        f.write(manual_sync)
    
    run_command('chmod +x setup-sync-command.sh', 'Created manual sync command setup')
    run_command('./setup-sync-command.sh', 'Setting up sync command')
    
    # 3. Create git hooks for automatic sync on commit
    git_hook = '''#!/bin/bash
# Post-commit hook - runs after every commit
# Automatically pushes to GitHub

cd /home/aadel/projects/22_MyAgent
git push origin master:main
echo "🚀 Auto-pushed to GitHub after commit"
'''

    hooks_dir = Path('.git/hooks')
    hooks_dir.mkdir(exist_ok=True)
    
    with open(hooks_dir / 'post-commit', 'w') as f:
        f.write(git_hook)
    
    run_command('chmod +x .git/hooks/post-commit', 'Created post-commit git hook')
    
    # 4. Create cron job for periodic sync
    cron_setup = '''#!/bin/bash
# Setup cron job for automatic syncing every 30 minutes during work hours
CRON_JOB="*/30 9-18 * * 1-5 cd /home/aadel/projects/22_MyAgent && ./auto-sync.sh >/dev/null 2>&1"

# Remove any existing auto-sync cron jobs
crontab -l 2>/dev/null | grep -v "auto-sync.sh" | crontab -

# Add new cron job
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "✅ Cron job added: Auto-sync every 30 minutes (weekdays 9-18)"
echo "📋 To view: crontab -l"
echo "📋 To disable: crontab -e and remove the auto-sync line"
'''

    with open('setup-auto-sync-cron.sh', 'w') as f:
        f.write(cron_setup)
    
    run_command('chmod +x setup-auto-sync-cron.sh', 'Created cron job setup script')
    
    # 5. Test GitHub connection
    run_command('gh auth status', 'Checking GitHub CLI authentication')
    
    # 6. Run initial sync
    print("\n🚀 Running initial sync to GitHub...")
    run_command('./auto-sync.sh', 'Initial GitHub sync')
    
    print("\n✅ GitHub Auto-Sync Setup Complete!")
    print("\n📋 Available commands:")
    print("   • ./auto-sync.sh - Manual sync anytime")
    print("   • sync - Quick sync command (after setup-sync-command.sh)")
    print("   • ./setup-auto-sync-cron.sh - Enable automatic periodic sync")
    print("\n🔧 Features enabled:")
    print("   • Git post-commit hook (auto-push after commits)")
    print("   • Manual auto-sync script")
    print("   • Shell alias for quick sync")
    print("   • Optional cron job for periodic sync")

if __name__ == "__main__":
    setup_github_autosync()
