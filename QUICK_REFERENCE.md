# MyAgent Quick Reference

## 🚀 Start Commands

### Development Mode
```bash
source venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### CLI Mode
```bash
python -m core --project "MyProject" --spec '{"description":"Your project"}'
```

### Docker
```bash
docker-compose up -d
```

## 🔑 Environment Variables

Required:
- `OPENAI_API_KEY` - Your OpenAI API key
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string

Optional:
- `ANTHROPIC_API_KEY` - For Claude models
- `DEFAULT_LLM_PROVIDER` - openai/anthropic/ollama
- `DEBUG` - Enable debug mode

## 📍 API Endpoints

### Health & Status
- `GET /health` - System health check
- `GET /api/v1/projects` - List projects
- `GET /api/v1/agents` - List agents

### Projects
- `POST /api/v1/projects` - Create project
- `GET /api/v1/projects/{id}` - Get project
- `POST /api/v1/projects/{id}/start` - Start development
- `POST /api/v1/projects/{id}/pause` - Pause
- `POST /api/v1/projects/{id}/resume` - Resume

### Metrics
- `GET /api/v1/projects/{id}/metrics` - Quality metrics
- `GET /api/v1/projects/{id}/iterations` - Iteration history

### WebSocket
- `WS /ws/projects/{id}` - Real-time updates

## 🔧 Common Tasks

### Run Tests
```bash
pytest tests/ -v
pytest tests/test_integration.py
```

### Database Setup
```bash
python scripts/setup_database.py
alembic upgrade head
```

### Check Logs
```bash
tail -f logs/orchestrator.log
tail -f logs/agents/*.log
```

### Reset Everything
```bash
python scripts/reset_database.py
rm -rf persistence/database/*.db
rm -rf persistence/snapshots/*
```

## 📊 Quality Metrics

System tracks 8 metrics:
1. Test Coverage (target: 95%)
2. Critical Bugs (target: 0)
3. Performance Score (target: 90%)
4. Documentation Coverage (target: 90%)
5. Code Quality Score (target: 85%)
6. Security Score (target: 95%)
7. Maintainability Index (target: 85%)
8. User Satisfaction (target: 90%)

## 🤖 Agent Capabilities

**CoderAgent**
- Code generation
- Refactoring
- Optimization

**TesterAgent**
- Test generation
- Test execution
- Coverage analysis

**DebuggerAgent**
- Error analysis
- Debugging
- Fix suggestions

**ArchitectAgent**
- Design review
- Pattern suggestions
- Scalability analysis

**AnalyzerAgent**
- Metric monitoring
- Trend analysis
- Performance tracking

**UIRefinerAgent**
- UI improvements
- UX optimization
- Accessibility checks

## 📂 Directory Structure

```
core/agents/        - Agent implementations
core/orchestrator/  - Main orchestrator
core/memory/        - Memory systems
api/                - REST API
config/             - Configuration
tests/              - Test suites
persistence/        - Data storage
logs/               - Application logs
```

## 🐛 Troubleshooting

**Import errors:** Check venv activated
**API won't start:** Check .env file
**Database errors:** Run setup_database.py
**Agent errors:** Check API keys set
**Tests fail:** Install test dependencies

## 💾 Backup & Recovery

### Create Checkpoint
Automatic every hour, manual via:
```python
await director.create_checkpoint()
```

### Restore
```python
director = ContinuousDirector.from_checkpoint(checkpoint_id)
```

## 🔐 Security

- API keys in .env (never commit!)
- JWT authentication enabled
- Rate limiting configured
- CORS properly set
- Input validation on all endpoints

## 📞 Getting Help

1. Check CLAUDE.md for detailed info
2. Review logs in logs/ directory
3. Run with --debug flag
4. Check GitHub issues

---
**Quick Start:** Set API key → Run `uvicorn api.main:app` → Create project
