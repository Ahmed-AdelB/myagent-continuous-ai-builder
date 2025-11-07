#!/bin/bash
# Setup cron job for automatic syncing every 30 minutes during work hours
CRON_JOB="*/30 9-18 * * 1-5 cd /home/aadel/projects/22_MyAgent && ./auto-sync.sh >/dev/null 2>&1"

# Remove any existing auto-sync cron jobs
crontab -l 2>/dev/null | grep -v "auto-sync.sh" | crontab -

# Add new cron job
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "✅ Cron job added: Auto-sync every 30 minutes (weekdays 9-18)"
echo "📋 To view: crontab -l"
echo "📋 To disable: crontab -e and remove the auto-sync line"
