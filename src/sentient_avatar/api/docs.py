import json
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

DESCRIPTION = (Path(__file__).resolve().parents[2] / "docs" / "api.md").read_text()


def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Sentient Avatar API",
        version="1.0.0",
        description=DESCRIPTION,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def generate_api_docs(app: FastAPI, output_dir: str = "docs") -> None:
    """Generate API documentation files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    openapi_schema = custom_openapi(app)

    with open(output_path / "openapi.json", "w") as f:
        json.dump(openapi_schema, f, indent=2)

    with open(output_path / "openapi.yaml", "w") as f:
        yaml.dump(openapi_schema, f, sort_keys=False)
