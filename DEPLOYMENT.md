# 🚀 Deployment Guide - MyAgent

Production deployment guide for MyAgent Continuous AI Builder.

---

## 🎯 Deployment Options

### Option 1: Docker Compose (Recommended)
### Option 2: Manual Deployment
### Option 3: Cloud Deployment (AWS/GCP/Azure)

---

## 📦 Option 1: Docker Compose (Easiest)

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum
- 20GB disk space

### Quick Deploy

```bash
# 1. Clone/navigate to project
cd /home/aadel/projects/22_MyAgent

# 2. Create .env file
cp .env.example .env
nano .env  # Add your API keys

# 3. Start all services
docker-compose --profile prod up -d

# 4. Check status
docker-compose ps

# 5. View logs
docker-compose logs -f
```

### Services Started
- PostgreSQL 15 (port 5432)
- Redis 7 (port 6379)
- ChromaDB (port 8000)
- MyAgent API (port 8001)
- Nginx (ports 80, 443)

### Access Points
- API: http://localhost/api
- Docs: http://localhost/api/docs
- Frontend: http://localhost
- Health: http://localhost/api/health

---

## 🛠️ Option 2: Manual Deployment

### Step 1: Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.11 python3.11-venv postgresql-15 redis-server nginx

# Create user
sudo useradd -m -s /bin/bash myagent
sudo su - myagent
```

### Step 2: Application Setup

```bash
# Clone code
git clone <your-repo> /home/myagent/22_MyAgent
cd /home/myagent/22_MyAgent

# Create venv
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Database Setup

```bash
# Create database
sudo -u postgres psql
CREATE DATABASE myagent_db;
CREATE USER myagent_user WITH PASSWORD 'strong_password_here';
GRANT ALL PRIVILEGES ON DATABASE myagent_db TO myagent_user;
\q

# Initialize schema
python scripts/setup_database.py
```

### Step 4: Configure Environment

```bash
# Create .env
cp .env.example .env
nano .env
```

**Required settings:**
```env
OPENAI_API_KEY=sk-proj-YOUR_KEY
DATABASE_URL=postgresql://myagent_user:strong_password_here@localhost/myagent_db
REDIS_URL=redis://localhost:6379
DEBUG=false
SECRET_KEY=generate-a-random-secret-key-here
```

### Step 5: Systemd Services

**API Service** (`/etc/systemd/system/myagent-api.service`):
```ini
[Unit]
Description=MyAgent API Server
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=myagent
WorkingDirectory=/home/myagent/22_MyAgent
Environment="PATH=/home/myagent/22_MyAgent/venv/bin"
ExecStart=/home/myagent/22_MyAgent/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Orchestrator Service** (`/etc/systemd/system/myagent-orchestrator.service`):
```ini
[Unit]
Description=MyAgent Orchestrator
After=network.target postgresql.service redis.service myagent-api.service

[Service]
Type=simple
User=myagent
WorkingDirectory=/home/myagent/22_MyAgent
Environment="PATH=/home/myagent/22_MyAgent/venv/bin"
ExecStart=/home/myagent/22_MyAgent/venv/bin/python -m core --project production --spec '{"description": "Production instance"}'
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable myagent-api myagent-orchestrator
sudo systemctl start myagent-api myagent-orchestrator
sudo systemctl status myagent-api myagent-orchestrator
```

### Step 6: Nginx Configuration

**`/etc/nginx/sites-available/myagent`:**
```nginx
upstream myagent_api {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 100M;

    # API
    location /api/ {
        proxy_pass http://myagent_api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket
    location /ws {
        proxy_pass http://myagent_api/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    # Frontend
    location / {
        root /home/myagent/22_MyAgent/frontend/dist;
        try_files $uri $uri/ /index.html;
    }
}
```

**Enable:**
```bash
sudo ln -s /etc/nginx/sites-available/myagent /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Step 7: SSL with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## ☁️ Option 3: Cloud Deployment

### AWS Deployment

**Using EC2 + RDS + ElastiCache:**

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04
   - Type: t3.medium (minimum)
   - Security Group: Ports 80, 443, 22

2. **Setup RDS PostgreSQL**
   - Engine: PostgreSQL 15
   - Instance: db.t3.micro
   - Update DATABASE_URL in .env

3. **Setup ElastiCache Redis**
   - Engine: Redis 7
   - Node: cache.t3.micro
   - Update REDIS_URL in .env

4. **Deploy Application**
   - Follow Manual Deployment steps
   - Use AWS Systems Manager for secrets

### GCP Deployment

**Using Cloud Run + Cloud SQL:**

```bash
# Build container
docker build -t gcr.io/YOUR_PROJECT/myagent:latest .

# Push to registry
docker push gcr.io/YOUR_PROJECT/myagent:latest

# Deploy
gcloud run deploy myagent \
  --image gcr.io/YOUR_PROJECT/myagent:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="DATABASE_URL=postgresql://..." \
  --memory 2Gi
```

### Azure Deployment

**Using App Service + Azure Database:**

```bash
# Create resource group
az group create --name myagent-rg --location eastus

# Create PostgreSQL
az postgres server create \
  --resource-group myagent-rg \
  --name myagent-db \
  --sku-name B_Gen5_1

# Create App Service
az webapp create \
  --resource-group myagent-rg \
  --plan myagent-plan \
  --name myagent-app \
  --runtime "PYTHON:3.11"

# Deploy code
az webapp deployment source config-zip \
  --resource-group myagent-rg \
  --name myagent-app \
  --src deploy.zip
```

---

## 🔒 Security Checklist

### Before Production
- [ ] Change all default passwords
- [ ] Generate strong SECRET_KEY
- [ ] Enable HTTPS/SSL
- [ ] Configure firewall (ufw/iptables)
- [ ] Setup fail2ban
- [ ] Enable database encryption
- [ ] Configure backup strategy
- [ ] Set up monitoring/alerting
- [ ] Review CORS settings
- [ ] Enable rate limiting
- [ ] Set up log rotation
- [ ] Configure security headers

### Secrets Management
```bash
# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Store in environment
echo "SECRET_KEY=<generated-key>" >> .env
chmod 600 .env
```

---

## 📊 Monitoring Setup

### Prometheus + Grafana

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  prometheus_data:
  grafana_data:
```

**Start:**
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

**Access:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

---

## 🔄 Backup Strategy

### Database Backups

**Daily backup script** (`/home/myagent/backup.sh`):
```bash
#!/bin/bash
BACKUP_DIR="/home/myagent/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup PostgreSQL
pg_dump -U myagent_user myagent_db | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# Backup persistence data
tar -czf $BACKUP_DIR/persistence_$DATE.tar.gz persistence/

# Keep only last 7 days
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete

echo "Backup completed: $DATE"
```

**Add to crontab:**
```bash
crontab -e
# Add: 0 2 * * * /home/myagent/backup.sh
```

---

## 🚨 Troubleshooting Production

### High Memory Usage
```bash
# Check processes
ps aux --sort=-%mem | head

# Restart services
sudo systemctl restart myagent-api myagent-orchestrator
```

### Database Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connections
sudo -u postgres psql -c "SELECT count(*) FROM pg_stat_activity;"

# Reset connections
sudo systemctl restart postgresql
```

### API Not Responding
```bash
# Check logs
journalctl -u myagent-api -n 100 --no-pager

# Check port
netstat -tlnp | grep 8000

# Restart
sudo systemctl restart myagent-api
```

---

## 📈 Scaling Production

### Horizontal Scaling

**Load Balancer Configuration:**
```nginx
upstream myagent_cluster {
    least_conn;
    server 10.0.1.10:8000 weight=1;
    server 10.0.1.11:8000 weight=1;
    server 10.0.1.12:8000 weight=1;
}
```

### Database Scaling
- Enable PostgreSQL replication
- Use connection pooling (PgBouncer)
- Implement read replicas
- Consider sharding for large datasets

### Redis Scaling
- Use Redis Cluster
- Enable persistence (AOF)
- Setup Redis Sentinel for HA

---

## ✅ Post-Deployment Checklist

- [ ] All services running
- [ ] Health check passing
- [ ] SSL certificate valid
- [ ] Backups configured
- [ ] Monitoring active
- [ ] Logs rotating
- [ ] Alerts configured
- [ ] Documentation updated
- [ ] Team trained
- [ ] Rollback plan ready

---

**You're ready for production!** 🚀
