# Installation Guide

This guide will help you set up the Sentient Avatar system on your local machine or server.

## Prerequisites

- Python 3.11 or higher
- PostgreSQL 16 or higher
- Redis 7 or higher
- Docker and Docker Compose (optional)

## System Requirements

- CPU: 2+ cores
- RAM: 4GB minimum, 8GB recommended
- Storage: 10GB minimum
- OS: Linux, macOS, or Windows 10/11

## Installation Methods

### 1. Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentient-avatar.git
   cd sentient-avatar
   ```

2. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Start the services:
   ```bash
   docker-compose up -d
   ```

4. Verify the installation:
   ```bash
   curl http://localhost:8000/health
   ```

### 2. Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentient-avatar.git
   cd sentient-avatar
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Initialize the database:
   ```bash
   alembic upgrade head
   ```

6. Start the application:
   ```bash
   uvicorn src.main:app --reload
   ```

## Post-Installation

1. Create an admin user:
   ```bash
   curl -X POST http://localhost:8000/api/v1/users/ \
     -H "Content-Type: application/json" \
     -d '{
       "email": "admin@example.com",
       "username": "admin",
       "password": "your-secure-password",
       "is_superuser": true
     }'
   ```

2. Access the API documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Verify PostgreSQL is running
   - Check database credentials in `.env`
   - Ensure database exists

2. **Redis Connection Error**
   - Verify Redis is running
   - Check Redis configuration in `.env`

3. **Port Conflicts**
   - Check if ports 8000, 5432, 6379 are available
   - Modify port configurations in `.env` if needed

### Getting Help

- Check the [FAQ](faq.md)
- Open an issue on GitHub
- Join our community chat

## Next Steps

- Read the [Configuration Guide](configuration.md)
- Follow the [Quick Start Guide](quick-start.md)
- Explore the [API Documentation](../api/overview.md) 