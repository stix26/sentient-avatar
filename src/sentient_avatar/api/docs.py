from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI
from typing import Dict, Any
import json
import yaml
from pathlib import Path

def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Sentient Avatar API",
        version="1.0.0",
        description="""
        # Sentient Avatar API Documentation
        
        This API provides endpoints for interacting with the Sentient Avatar system, including:
        
        - Chat and conversation management
        - Audio processing and speech synthesis
        - Avatar video generation
        - Image analysis and vision processing
        - User management and authentication
        - System monitoring and metrics
        
        ## Authentication
        
        All endpoints require authentication using JWT tokens. Include the token in the Authorization header:
        
        ```
        Authorization: Bearer <your-token>
        ```
        
        ## Rate Limiting
        
        API requests are rate-limited to prevent abuse. The current limits are:
        
        - 100 requests per minute for authenticated users
        - 10 requests per minute for unauthenticated users
        
        ## Error Handling
        
        The API uses standard HTTP status codes and returns detailed error messages in the response body.
        
        ## WebSocket Support
        
        Real-time communication is supported through WebSocket connections for:
        
        - Live chat
        - Audio streaming
        - Video streaming
        - Real-time avatar updates
        
        ## Security
        
        - All endpoints are protected by JWT authentication
        - Rate limiting is enforced
        - Input validation is performed
        - CORS is configured
        - MFA is supported for sensitive operations
        """,
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter your JWT token in the format: Bearer <token>"
        },
        "apiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "Enter your API key"
        }
    }
    
    # Add global security requirement
    openapi_schema["security"] = [{"bearerAuth": []}]
    
    # Add tags
    openapi_schema["tags"] = [
        {
            "name": "Authentication",
            "description": "User authentication and authorization endpoints"
        },
        {
            "name": "Chat",
            "description": "Chat and conversation management endpoints"
        },
        {
            "name": "Audio",
            "description": "Audio processing and speech synthesis endpoints"
        },
        {
            "name": "Avatar",
            "description": "Avatar video generation and management endpoints"
        },
        {
            "name": "Vision",
            "description": "Image analysis and vision processing endpoints"
        },
        {
            "name": "Users",
            "description": "User management endpoints"
        },
        {
            "name": "System",
            "description": "System monitoring and metrics endpoints"
        }
    ]
    
    # Add examples
    openapi_schema["components"]["examples"] = {
        "ChatRequest": {
            "value": {
                "text": "Hello, how are you?",
                "context": {
                    "previous_messages": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello! How can I help you?"}
                    ]
                }
            }
        },
        "ChatResponse": {
            "value": {
                "text": "I'm doing well, thank you for asking! How can I assist you today?",
                "audio": "base64-encoded-audio-data",
                "video": "base64-encoded-video-data"
            }
        },
        "ErrorResponse": {
            "value": {
                "error": "Invalid input",
                "detail": "The provided input does not meet the requirements",
                "code": "INVALID_INPUT"
            }
        }
    }
    
    # Add response schemas
    openapi_schema["components"]["schemas"].update({
        "Error": {
            "type": "object",
            "properties": {
                "error": {"type": "string"},
                "detail": {"type": "string"},
                "code": {"type": "string"}
            }
        },
        "Success": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "data": {"type": "object"}
            }
        }
    })
    
    # Add WebSocket documentation
    openapi_schema["components"]["schemas"]["WebSocketMessage"] = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["text", "audio", "image", "control"]
            },
            "data": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "audio": {"type": "string", "format": "binary"},
                    "image": {"type": "string", "format": "binary"},
                    "control": {"type": "object"}
                }
            }
        }
    }
    
    # Add rate limit headers
    openapi_schema["components"]["headers"] = {
        "X-RateLimit-Limit": {
            "description": "The maximum number of requests allowed per time window",
            "schema": {"type": "integer"}
        },
        "X-RateLimit-Remaining": {
            "description": "The number of requests remaining in the current time window",
            "schema": {"type": "integer"}
        },
        "X-RateLimit-Reset": {
            "description": "The time at which the current rate limit window resets",
            "schema": {"type": "string", "format": "date-time"}
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

def generate_api_docs(app: FastAPI, output_dir: str = "docs") -> None:
    """Generate API documentation files."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate OpenAPI schema
    openapi_schema = custom_openapi(app)
    
    # Save as JSON
    with open(output_path / "openapi.json", "w") as f:
        json.dump(openapi_schema, f, indent=2)
    
    # Save as YAML
    with open(output_path / "openapi.yaml", "w") as f:
        yaml.dump(openapi_schema, f, sort_keys=False)
    
    # Generate HTML documentation
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentient Avatar API Documentation</title>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
        <style>
            body {
                margin: 0;
                padding: 0;
                font-family: 'Roboto', sans-serif;
                line-height: 1.6;
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            h1, h2, h3 {
                font-family: 'Montserrat', sans-serif;
                color: #2c3e50;
            }
            pre {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }
            code {
                font-family: 'Courier New', monospace;
                background-color: #f8f9fa;
                padding: 2px 5px;
                border-radius: 3px;
            }
            .endpoint {
                margin-bottom: 30px;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .method {
                font-weight: bold;
                color: #e74c3c;
            }
            .path {
                font-family: 'Courier New', monospace;
                color: #3498db;
            }
            .description {
                margin: 10px 0;
            }
            .parameters, .responses {
                margin-top: 15px;
            }
            .parameter, .response {
                margin: 10px 0;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 3px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Sentient Avatar API Documentation</h1>
            <p>Version: {version}</p>
            <div class="description">
                {description}
            </div>
            <h2>Endpoints</h2>
            {endpoints}
        </div>
    </body>
    </html>
    """
    
    # Generate endpoints documentation
    endpoints_html = ""
    for path, path_item in openapi_schema["paths"].items():
        for method, operation in path_item.items():
            endpoints_html += f"""
            <div class="endpoint">
                <h3><span class="method">{method.upper()}</span> <span class="path">{path}</span></h3>
                <div class="description">{operation.get('description', '')}</div>
                <div class="parameters">
                    <h4>Parameters</h4>
                    {_generate_parameters_html(operation.get('parameters', []))}
                </div>
                <div class="responses">
                    <h4>Responses</h4>
                    {_generate_responses_html(operation.get('responses', {}))}
                </div>
            </div>
            """
    
    # Generate HTML documentation
    html_content = html_template.format(
        version=openapi_schema["info"]["version"],
        description=openapi_schema["info"]["description"],
        endpoints=endpoints_html
    )
    
    with open(output_path / "index.html", "w") as f:
        f.write(html_content)

def _generate_parameters_html(parameters: list) -> str:
    """Generate HTML for parameters documentation."""
    html = ""
    for param in parameters:
        html += f"""
        <div class="parameter">
            <strong>{param['name']}</strong> ({param['in']}) - {param.get('description', '')}
            <br>
            Type: {param['schema']['type']}
            {f"<br>Required: {param['required']}" if 'required' in param else ''}
        </div>
        """
    return html

def _generate_responses_html(responses: dict) -> str:
    """Generate HTML for responses documentation."""
    html = ""
    for status_code, response in responses.items():
        html += f"""
        <div class="response">
            <strong>{status_code}</strong> - {response.get('description', '')}
            {_generate_schema_html(response.get('content', {}).get('application/json', {}).get('schema', {}))}
        </div>
        """
    return html

def _generate_schema_html(schema: dict) -> str:
    """Generate HTML for schema documentation."""
    if not schema:
        return ""
    
    html = "<pre>"
    html += json.dumps(schema, indent=2)
    html += "</pre>"
    return html 