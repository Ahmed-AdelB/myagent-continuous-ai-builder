#!/bin/bash
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
