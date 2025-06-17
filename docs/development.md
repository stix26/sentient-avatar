# Development Guide

## Overview

This guide provides information for developers who want to contribute to the Sentient Avatar project. It covers the development environment setup, code structure, testing, and best practices.

## Development Environment

### Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- Docker and Docker Compose
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sentient-avatar.git
cd sentient-avatar
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Project Structure

```
sentient-avatar/
├── src/
│   └── sentient_avatar/
│       ├── services/          # Service connections
│       ├── frontend/          # Frontend code
│       └── avatar_server.py   # FastAPI server
├── tests/                     # Test files
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
└── docker/                    # Docker configuration
```

## Code Style

- Follow PEP 8 for Python code
- Use type hints for all function parameters and return values
- Write docstrings for all modules, classes, and functions
- Use meaningful variable and function names
- Keep functions small and focused
- Use async/await for I/O operations

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_services.py

# Run tests with coverage
pytest --cov=sentient_avatar

# Run tests in parallel
pytest -n auto
```

### Writing Tests

- Write tests for all new features
- Use fixtures for common test setup
- Mock external services
- Test both success and error cases
- Use descriptive test names

Example:
```python
@pytest.mark.asyncio
async def test_llm_service_generate():
    service = LLMService()
    result = await service.generate("Hello")
    assert result["text"] is not None
```

## Adding New Features

1. Create a new branch:
```bash
git checkout -b feature/new-feature
```

2. Implement the feature:
   - Add new code
   - Write tests
   - Update documentation

3. Run tests and linting:
```bash
pytest
flake8
mypy
```

4. Create a pull request:
   - Describe the changes
   - Link related issues
   - Request review

## Service Development

### Adding a New Service

1. Create a new service class in `src/sentient_avatar/services/`:
```python
from .base import BaseService

class NewService(BaseService):
    def __init__(self):
        super().__init__()
        self.service_url = "http://new-service:8000"

    async def health_check(self):
        # Implement health check
        pass

    async def initialize(self):
        # Implement initialization
        pass
```

2. Add service configuration to `docker-compose.yml`:
```yaml
services:
  new-service:
    build: ./docker/new-service
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
```

3. Update the FastAPI server to use the new service:
```python
from sentient_avatar.services.new_service import NewService

app = FastAPI()
new_service = NewService()

@app.on_event("startup")
async def startup():
    await new_service.initialize()
```

## Frontend Development

### SillyTavern Extension

1. Development setup:
```bash
cd src/sentient_avatar/frontend
npm install
npm run dev
```

2. Building:
```bash
npm run build
```

3. Testing:
```bash
npm test
```

## Debugging

### Logging

- Use the built-in logging module
- Set appropriate log levels
- Include relevant context in log messages

Example:
```python
import logging

logger = logging.getLogger(__name__)

def some_function():
    logger.debug("Starting function")
    try:
        # Do something
        logger.info("Operation successful")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
```

### Debugging Tools

- Use `pdb` for Python debugging
- Use browser developer tools for frontend debugging
- Use Docker logs for service debugging

## Performance Optimization

1. Profile the code:
```bash
python -m cProfile -o output.prof script.py
```

2. Analyze the results:
```bash
python -m pstats output.prof
```

3. Common optimizations:
   - Use connection pooling
   - Implement caching
   - Optimize database queries
   - Use async I/O
   - Implement rate limiting

## Security

1. Input validation:
   - Validate all user input
   - Use type hints
   - Implement proper error handling

2. Authentication:
   - Implement proper authentication
   - Use secure password hashing
   - Implement rate limiting

3. Data protection:
   - Encrypt sensitive data
   - Use secure communication
   - Implement proper access control

## Deployment

1. Build Docker images:
```bash
docker-compose build
```

2. Run services:
```bash
docker-compose up -d
```

3. Monitor services:
```bash
docker-compose logs -f
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests
5. Create a pull request

## Resources

- [Python Documentation](https://docs.python.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [SillyTavern Documentation](https://github.com/SillyTavern/SillyTavern)
- [CrewAI Documentation](https://github.com/joaomdmoura/crewAI)
- [AutoGen Documentation](https://github.com/microsoft/autogen) 