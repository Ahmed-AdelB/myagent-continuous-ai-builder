# 🚀 START HERE - Quick Launch Guide

**Welcome to MyAgent!** This is your autonomous AI development system.

---

## ⚡ FASTEST START (3 Commands)

```bash
cd /home/aadel/projects/22_MyAgent
./quick_start.sh
python -m core --project myapp --spec '{"description": "My app"}'
```

**Done!** Your autonomous AI agent is now working!

---

## 📋 WHAT YOU NEED

**Before starting:**
1. ✅ Python 3.11+ installed
2. ✅ Virtual environment created (already done)
3. ⚠️ API Keys (OpenAI or Anthropic) - **REQUIRED**
4. ⚠️ PostgreSQL running (or use Docker)

---

## 🔐 STEP 1: API Keys (5 minutes)

**CRITICAL**: You posted an API key publicly - it's compromised!

### Revoke Old Key:
1. Go to https://platform.openai.com/api-keys
2. Find and DELETE the key starting with `sk-proj-HbzTHqDFyxO5tQ...`

### Get New Keys:
- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/settings/keys

### Add to .env:
```bash
nano .env
```

Change this line:
```
OPENAI_API_KEY=your-openai-key-here
```

To your real key:
```
OPENAI_API_KEY=sk-proj-YOUR_NEW_KEY_HERE
```

Or use local Ollama (no key needed):
```
DEFAULT_LLM_PROVIDER=ollama
```

---

## 🚀 STEP 2: Run Setup (5 minutes)

```bash
./quick_start.sh
```

This will:
- ✅ Activate virtual environment
- ✅ Fix dependencies
- ✅ Test imports
- ✅ Setup database
- ✅ Verify everything works

---

## 🎯 STEP 3: Start Your First Project (1 minute)

```bash
python -m core \
  --project my-first-app \
  --spec '{
    "description": "A simple hello world web app",
    "requirements": [
      "Create a FastAPI server",
      "Add a hello world endpoint",
      "Write tests"
    ]
  }'
```

**Watch it work!** It will:
- Initialize 6 AI agents
- Start generating code
- Write tests automatically
- Debug any failures
- Keep improving until perfect!

Press Ctrl+C to stop gracefully.

---

## 🌐 STEP 4: Start API Server (Optional)

In a **new terminal**:

```bash
cd /home/aadel/projects/22_MyAgent
source venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

Now you can:
- View API docs: http://localhost:8000/docs
- Check health: http://localhost:8000/health
- Create projects via REST API
- Monitor via WebSocket

---

## 📊 STEP 5: Monitor Progress

### View Logs:
```bash
tail -f logs/orchestrator.log
```

### Check Metrics:
```bash
curl http://localhost:8000/projects/{project_id}/metrics
```

### View Dashboard:
```bash
cd frontend && npm run dev
```
Then open: http://localhost:5173

---

## 🎓 ADVANCED: Make MyAgent Improve Itself

Once it's running, make it work on its own codebase:

```bash
curl -X POST http://localhost:8000/projects \
  -H "Content-Type: application/json" \
  -d '{
    "name": "self-improvement",
    "description": "Make MyAgent better",
    "requirements": [
      "Increase test coverage to 95%",
      "Fix all bugs",
      "Optimize performance"
    ],
    "max_iterations": 1000
  }'
```

**It will autonomously improve itself!** 🤯

---

## ❓ IF SOMETHING BREAKS

### Import Errors:
```bash
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

### Database Errors:
```bash
docker-compose up -d postgres redis
python scripts/setup_database.py
```

### Can't Find Python Module:
```bash
export PYTHONPATH=/home/aadel/projects/22_MyAgent:$PYTHONPATH
```

### Still Broken?
Read `FINAL_STATUS.md` for detailed troubleshooting.

---

## 📚 MORE INFORMATION

- **FINAL_STATUS.md** - Complete system status and how it works
- **NIGHT_HANDOFF.md** - Detailed setup instructions
- **CLAUDE.md** - Full architecture documentation
- **quick_start.sh** - Automated setup script

---

## 🎉 YOU'RE READY!

Your autonomous AI development system is ready to:
- ✅ Build applications continuously
- ✅ Never give up until perfect
- ✅ Learn from mistakes
- ✅ Work 24/7 without you

**Just run the 3 commands at the top and watch it work!**

---

**Time to First Running System**: 10-30 minutes
**Difficulty**: Easy (automated setup)
**Support**: Read FINAL_STATUS.md if issues

## GO BUILD SOMETHING AMAZING! 🚀
