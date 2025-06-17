# Configuration Guide

This guide explains how to configure the Sentient Avatar system to meet your specific needs.

## Environment Variables

The system uses environment variables for configuration. Copy `.env.example` to `.env` and modify the values:

```bash
cp .env.example .env
```

### Essential Configuration

#### Application
```env
APP_ENV=development  # development, testing, production
DEBUG=true  # Set to false in production
API_V1_STR=/api/v1
PROJECT_NAME=Sentient Avatar
VERSION=0.1.0
```

#### Security
```env
SECRET_KEY=your-secret-key-here  # Change in production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
```

#### Database
```env
DATABASE_URL=postgresql://user:password@localhost:5432/sentient_avatar
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10
```

#### Redis
```env
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=
REDIS_SSL=false
```

### Optional Configuration

#### Monitoring
```env
ENABLE_METRICS=true
ENABLE_TRACING=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

#### Logging
```env
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json  # json, text
LOG_FILE=logs/app.log
```

#### Rate Limiting
```env
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
```

## Configuration Files

### 1. Database Migrations (Alembic)

Located in `alembic/`:
- `alembic.ini`: Main configuration
- `env.py`: Environment setup
- `versions/`: Migration scripts

### 2. Monitoring (Prometheus)

Located in `prometheus.yml`:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'sentient-avatar'
    static_configs:
      - targets: ['api:8000']
```

### 3. Alert Rules

Located in `alert_rules.yml`:
```yaml
groups:
  - name: sentient-avatar
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
```

## Production Configuration

### Security Best Practices

1. Use strong secrets:
   ```env
   SECRET_KEY=$(openssl rand -hex 32)
   ```

2. Enable SSL/TLS:
   ```env
   REDIS_SSL=true
   DATABASE_SSL=true
   ```

3. Set appropriate timeouts:
   ```env
   DATABASE_POOL_TIMEOUT=30
   DATABASE_POOL_RECYCLE=1800
   ```

### Performance Tuning

1. Database connection pool:
   ```env
   DATABASE_POOL_SIZE=20
   DATABASE_MAX_OVERFLOW=30
   ```

2. Worker configuration:
   ```env
   WORKERS=4
   WORKER_CONNECTIONS=1000
   ```

3. Cache settings:
   ```env
   CACHE_TTL=300
   CACHE_PREFIX=sentient_avatar
   ```

## Configuration Validation

The system validates configuration on startup. Common validation errors:

1. **Invalid Database URL**
   - Check format: `postgresql://user:password@host:port/dbname`
   - Verify credentials

2. **Invalid Redis URL**
   - Check format: `redis://user:password@host:port/db`
   - Verify connection

3. **Invalid Secret Key**
   - Must be at least 32 characters
   - Use `openssl rand -hex 32` to generate

## Dynamic Configuration

Some settings can be updated at runtime:

1. Log level:
   ```bash
   curl -X POST http://localhost:8000/api/v1/admin/config/log-level \
     -H "Content-Type: application/json" \
     -d '{"level": "DEBUG"}'
   ```

2. Rate limits:
   ```bash
   curl -X POST http://localhost:8000/api/v1/admin/config/rate-limit \
     -H "Content-Type: application/json" \
     -d '{"per_minute": 100, "per_hour": 1000}'
   ```

## Next Steps

- Read the [Quick Start Guide](quick-start.md)
- Explore the [API Documentation](../api/overview.md)
- Check the [Monitoring Guide](../monitoring/overview.md) 