#!/bin/bash
# GitHub Repository Setup Script for 22_MyAgent

echo '🚀 22_MyAgent GitHub Integration Setup'
echo ''

# Check if already authenticated
if gh auth status >/dev/null 2>&1; then
    echo '✅ Already authenticated with GitHub'
else
    echo '❌ Not authenticated. Please run: gh auth login --with-token'
    echo 'Then paste your Personal Access Token'
    exit 1
fi

# Option 1: Create new repository
echo '📝 Choose an option:'
echo '1. Create new GitHub repository'
echo '2. Connect to existing repository'
read -p 'Enter choice (1 or 2): ' choice

if [ "$choice" = "1" ]; then
    echo ''
    echo '🆕 Creating new GitHub repository...'
    
    # Create repository
    gh repo create 22_MyAgent --public --description "Continuous AI App Builder - Multi-agent system that never stops until perfection" --clone=false
    
    # Add remote
    git remote add origin https://github.com/$(gh api user --jq .login)/22_MyAgent.git
    
    # Initial commit
    git add .
    git commit -m "Initial commit: 22_MyAgent Continuous AI App Builder
    
🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
    
    # Push to GitHub
    git push -u origin master
    
    echo ''
    echo '🎉 Repository created successfully!'
    echo "🔗 URL: https://github.com/$(gh api user --jq .login)/22_MyAgent"

elif [ "$choice" = "2" ]; then
    echo ''
    read -p 'Enter your GitHub username: ' username
    read -p 'Enter repository name (or press Enter for 22_MyAgent): ' reponame
    reponame=${reponame:-22_MyAgent}
    
    echo ''
    echo "🔗 Connecting to existing repository: $username/$reponame"
    
    # Add remote
    git remote add origin https://github.com/$username/$reponame.git
    
    # Fetch and merge
    git fetch origin
    git branch --set-upstream-to=origin/master master
    git pull origin master --allow-unrelated-histories
    
    # Commit current changes
    git add .
    git commit -m "Merge local 22_MyAgent with GitHub repository
    
🤖 Generated with Claude Code  
Co-Authored-By: Claude <noreply@anthropic.com>"
    
    # Push to GitHub
    git push origin master
    
    echo ''
    echo '🎉 Successfully connected to existing repository!'
    echo "🔗 URL: https://github.com/$username/$reponame"
else
    echo '❌ Invalid choice'
    exit 1
fi

echo ''
echo '✅ GitHub integration complete!'
echo '🚀 Your Continuous AI App Builder is now on GitHub!'
