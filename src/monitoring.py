from prometheus_client import Counter, Histogram, Gauge
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from src.config import settings

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests in progress',
    ['method', 'endpoint']
)

AVATAR_CREATIONS = Counter(
    'avatar_creations_total',
    'Total number of avatar creations'
)

AVATAR_UPDATES = Counter(
    'avatar_updates_total',
    'Total number of avatar updates'
)

EMOTION_CHANGES = Counter(
    'emotion_changes_total',
    'Total number of emotion changes',
    ['emotion']
)

COGNITIVE_PROCESSING_TIME = Histogram(
    'cognitive_processing_seconds',
    'Time spent on cognitive processing',
    ['operation']
)

PHYSICAL_ACTION_TIME = Histogram(
    'physical_action_seconds',
    'Time spent on physical actions',
    ['action']
)

# OpenTelemetry setup
def setup_tracing(app):
    if settings.ENABLE_TRACING:
        trace.set_tracer_provider(TracerProvider())
        otlp_exporter = OTLPSpanExporter(
            endpoint=f"{settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT}"
        )
        span_processor = BatchSpanProcessor(otlp_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        FastAPIInstrumentor.instrument_app(app)

# Example usage:
# from src.monitoring import REQUEST_COUNT, REQUEST_LATENCY
# 
# @app.middleware("http")
# async def monitor_requests(request: Request, call_next):
#     start_time = time.time()
#     response = await call_next(request)
#     duration = time.time() - start_time
#     
#     REQUEST_COUNT.labels(
#         method=request.method,
#         endpoint=request.url.path,
#         status=response.status_code
#     ).inc()
#     
#     REQUEST_LATENCY.labels(
#         method=request.method,
#         endpoint=request.url.path
#     ).observe(duration)
#     
#     return response 