from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from prometheus_client import make_asgi_app
from src.config import settings
from src.database import init_db
from src.logger import logger
from src.monitoring import setup_tracing, REQUEST_COUNT, REQUEST_LATENCY, ACTIVE_REQUESTS
from src.security import rate_limit_middleware

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.middleware("http")(rate_limit_middleware)

# Add monitoring middleware
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    
    # Increment active requests
    ACTIVE_REQUESTS.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    finally:
        # Decrement active requests
        ACTIVE_REQUESTS.labels(
            method=request.method,
            endpoint=request.url.path
        ).dec()

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Setup OpenTelemetry tracing
setup_tracing(app)

# Initialize database
@app.on_event("startup")
async def startup_event():
    try:
        init_db()
    except Exception as e:  # pragma: no cover - optional DB in tests
        logger.error(f"Database initialization failed: {e}")
    logger.info("Application startup complete")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Import and include routers
from src.api.v1.endpoints import auth, users, avatar

app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(users.router, prefix=settings.API_V1_STR)
app.include_router(avatar.router, prefix=settings.API_V1_STR)

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.DEBUG
    ) 