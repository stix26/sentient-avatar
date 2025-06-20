# Sentient Avatar

A sophisticated avatar system with emotional intelligence, capable of understanding and responding to user interactions in a natural and engaging way.

## 🌟 Features

- 🤖 **Emotional Intelligence**: Avatars can understand and express emotions
- 🧠 **Cognitive Processing**: Advanced decision-making and response generation
- 💪 **Physical Actions**: Natural movements and expressions
- 🔄 **Real-time Streaming**: Live updates of avatar states and behaviors
- 🔒 **Secure Authentication**: JWT-based authentication system
- 📊 **Monitoring**: Comprehensive metrics and logging
- 🚀 **Scalable Architecture**: Built with FastAPI and modern Python practices

## 🏗️ Architecture

The system is built with a modular architecture:

- **FastAPI**: Modern, fast web framework for building APIs
- **SQLAlchemy**: Powerful ORM for database operations
- **Redis**: High-performance caching and message broker
- **PostgreSQL**: Robust relational database
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Data visualization and dashboards
- **OpenTelemetry**: Distributed tracing and observability
- **RabbitMQ**: Message queue for async operations

## 🚀 Quick Start

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
# Edit `.env` with your local settings. The provided example uses a local
# PostgreSQL and Redis instance. Ensure the `POSTGRES_USER` in `.env` refers to
# an existing database role. The provided `.env.example` sets this to `postgres`.
# Using an invalid role such as `root` will prevent the application from connecting.
# If the role specified in `POSTGRES_USER` does not exist, create it first:
# `createuser -s $POSTGRES_USER`.
# When the database or Redis services are unavailable the application will start
# with database initialization errors logged but will continue running.

# Initialize database
alembic upgrade head

# Start the application
uvicorn src.main:app --reload
```

## 📚 Documentation

- [Installation Guide](docs/getting-started/installation.md)
- [Configuration Guide](docs/getting-started/configuration.md)
- [Quick Start Guide](docs/getting-started/quick-start.md)
- [API Documentation](docs/api/overview.md)
- [Architecture Guide](docs/architecture/overview.md)

## 🛠️ Development

### Prerequisites

- Python 3.11+
- PostgreSQL 16+
- Redis 7+
- Docker and Docker Compose (optional)

### Development Setup

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

3. Run tests:
   ```bash
   make test
   ```

4. Check code style:
   ```bash
   make lint
   ```

### Common Tasks

```bash
# Run development server
make dev

# Run tests
make test

# Format code
make format

# Check code style
make lint

# Clean up
make clean

# Build documentation
make docs

# Docker commands
make docker-build
make docker-up
make docker-down
```

## 📊 Monitoring

The system includes comprehensive monitoring:

- **Prometheus**: Metrics collection at `/metrics`
- **Grafana**: Dashboards at `http://localhost:3000`
- **OpenTelemetry**: Distributed tracing
- **Structured Logging**: JSON-formatted logs

### Redis Memory Overcommit

In production you may see warnings about memory overcommit when Redis starts.
Consider adding the following to your system configuration:

```bash
sudo sysctl vm.overcommit_memory=1
```

This setting allows Redis to allocate memory more effectively. Persist it by
adding `vm.overcommit_memory = 1` to `/etc/sysctl.conf` and reloading with
`sudo sysctl -p`.

## 🔒 Security

- JWT-based authentication
- Role-based access control
- Rate limiting
- Input validation
- Secure password hashing
- CORS protection
- Security headers
- Dependency scanning

## 🧪 Testing

- Unit tests with pytest
- Integration tests
- End-to-end tests
- Code coverage reporting
- Performance testing
- Security testing

## 📦 Project Structure

```
sentient-avatar/
├── src/                    # Source code
│   ├── api/               # API endpoints
│   ├── core/              # Core functionality
│   ├── db/                # Database models
│   ├── services/          # Business logic
│   └── utils/             # Utility functions
├── tests/                 # Test suite
├── docs/                  # Documentation
├── alembic/               # Database migrations
├── .github/               # GitHub configurations
├── docker/                # Docker configurations
└── scripts/               # Utility scripts
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- SQLAlchemy for the powerful ORM
- Pydantic for data validation
- Alembic for database migrations
- Prometheus and Grafana for monitoring
- OpenTelemetry for distributed tracing

## 📞 Support

- [GitHub Issues](https://github.com/yourusername/sentient-avatar/issues)
- [Documentation](https://yourusername.github.io/sentient-avatar)
- [Community Chat](https://discord.gg/your-server) 