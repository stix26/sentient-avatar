# Quick Start Guide

This guide will help you get started with the Sentient Avatar system quickly.

## Prerequisites

- Python 3.11+
- PostgreSQL 16+
- Redis 7+
- Docker and Docker Compose (optional)

## 1. Installation

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/sentient-avatar.git
cd sentient-avatar

# Start the services
docker-compose up -d
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sentient-avatar.git
cd sentient-avatar

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Initialize database
alembic upgrade head

# Start the application
uvicorn src.main:app --reload
```

## 2. Create Your First Avatar

1. Register a user:
   ```bash
   curl -X POST http://localhost:8000/api/v1/auth/register \
     -H "Content-Type: application/json" \
     -d '{
       "email": "user@example.com",
       "username": "user",
       "password": "your-password"
     }'
   ```

2. Login to get access token:
   ```bash
   curl -X POST http://localhost:8000/api/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{
       "username": "user@example.com",
       "password": "your-password"
     }'
   ```

3. Create an avatar:
   ```bash
   curl -X POST http://localhost:8000/api/v1/avatars \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "My Avatar",
       "description": "A friendly avatar",
       "personality": {
         "traits": ["friendly", "helpful", "curious"],
         "interests": ["technology", "art", "music"]
       },
       "appearance": {
         "style": "realistic",
         "features": {
           "hair": "brown",
           "eyes": "blue",
           "height": "average"
         }
       }
     }'
   ```

## 3. Interact with Your Avatar

1. Get avatar details:
   ```bash
   curl http://localhost:8000/api/v1/avatars/{avatar_id} \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
   ```

2. Stream avatar updates:
   ```bash
   curl http://localhost:8000/api/v1/avatars/{avatar_id}/stream \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
   ```

3. Update avatar state:
   ```bash
   curl -X PATCH http://localhost:8000/api/v1/avatars/{avatar_id} \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "current_emotion": "happy",
       "current_cognitive_state": "focused"
     }'
   ```

## 4. Monitor Your Avatar

1. View metrics:
   ```bash
   curl http://localhost:8000/metrics
   ```

2. Check logs:
   ```bash
   tail -f logs/app.log
   ```

3. Access Grafana dashboard:
   - Open http://localhost:3000
   - Login with admin/admin
   - Navigate to Sentient Avatar dashboard

## 5. Development Workflow

1. Run tests:
   ```bash
   pytest
   ```

2. Check code style:
   ```bash
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   ```

3. Generate documentation:
   ```bash
   mkdocs serve
   ```

## Common Tasks

### Update Avatar Personality

```bash
curl -X PATCH http://localhost:8000/api/v1/avatars/{avatar_id} \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "personality": {
      "traits": ["friendly", "helpful", "curious", "creative"],
      "interests": ["technology", "art", "music", "science"]
    }
  }'
```

### Change Avatar Appearance

```bash
curl -X PATCH http://localhost:8000/api/v1/avatars/{avatar_id} \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "appearance": {
      "style": "cartoon",
      "features": {
        "hair": "blue",
        "eyes": "green",
        "height": "tall"
      }
    }
  }'
```

### Set Avatar Voice

```bash
curl -X PATCH http://localhost:8000/api/v1/avatars/{avatar_id} \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "voice": {
      "type": "female",
      "accent": "british",
      "pitch": 1.2
    }
  }'
```

## Next Steps

- Read the [User Guide](../user-guide/authentication.md)
- Explore the [API Documentation](../api/overview.md)
- Check the [Architecture Guide](../architecture/overview.md) 