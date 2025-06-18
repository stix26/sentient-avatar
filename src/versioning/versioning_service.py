import os
import json
import time
import uuid
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge
import mlflow
from mlflow.tracking import MlflowClient
import redis
import ray
from ray import serve
from ray.serve.deployment import Deployment
from ray.serve.config import HTTPOptions
import numpy as np
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
MODEL_VERSION_COUNTER = Counter(
    "model_version_total", "Total number of model versions", ["model_name"]
)
AB_TEST_REQUESTS = Counter(
    "ab_test_requests_total", "Total number of A/B test requests", ["experiment_name"]
)
MODEL_PERFORMANCE = Gauge(
    "model_performance_metric",
    "Model performance metrics",
    ["model_version", "metric_name"],
)
EXPERIMENT_DURATION = Histogram(
    "experiment_duration_seconds", "Duration of A/B tests", ["experiment_name"]
)


class ModelVersion:
    def __init__(self, model_path: str, metadata: Dict[str, Any]):
        self.version_id = str(uuid.uuid4())
        self.model_path = model_path
        self.metadata = metadata
        self.created_at = datetime.now()
        self.performance_metrics = {}
        self.is_active = False
        self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of model files."""
        sha256_hash = hashlib.sha256()
        for root, _, files in os.walk(self.model_path):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


class ABTest:
    def __init__(
        self,
        name: str,
        model_a: ModelVersion,
        model_b: ModelVersion,
        config: Dict[str, Any],
    ):
        self.experiment_id = str(uuid.uuid4())
        self.name = name
        self.model_a = model_a
        self.model_b = model_b
        self.config = config
        self.start_time = datetime.now()
        self.end_time = None
        self.results = {
            "model_a": {"requests": 0, "successes": 0, "metrics": {}},
            "model_b": {"requests": 0, "successes": 0, "metrics": {}},
        }
        self.is_active = True


class ModelVersioningService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = redis.Redis(
            host=config["redis_host"], port=config["redis_port"], db=0
        )
        self.mlflow_client = MlflowClient()
        self.model_versions: Dict[str, ModelVersion] = {}
        self.active_experiments: Dict[str, ABTest] = {}

        # Initialize FastAPI app
        self.app = FastAPI()
        self._setup_routes()

    def _setup_routes(self):
        """Set up FastAPI routes."""

        class VersionRequest(BaseModel):
            model_path: str
            metadata: Dict[str, Any]

        class ABTestRequest(BaseModel):
            name: str
            model_a_version: str
            model_b_version: str
            config: Dict[str, Any]

        @self.app.post("/version")
        async def create_version(request: VersionRequest):
            try:
                version = ModelVersion(request.model_path, request.metadata)
                self.model_versions[version.version_id] = version

                # Log to MLflow
                self._log_version_to_mlflow(version)

                # Update metrics
                MODEL_VERSION_COUNTER.labels(
                    version.metadata.get("name", "unknown")
                ).inc()

                return {
                    "status": "success",
                    "version_id": version.version_id,
                    "created_at": version.created_at.isoformat(),
                }
            except Exception as e:
                logger.error(f"Error creating version: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/ab-test")
        async def create_ab_test(request: ABTestRequest):
            try:
                model_a = self.model_versions.get(request.model_a_version)
                model_b = self.model_versions.get(request.model_b_version)

                if not model_a or not model_b:
                    raise HTTPException(
                        status_code=404, detail="One or both model versions not found"
                    )

                experiment = ABTest(request.name, model_a, model_b, request.config)

                self.active_experiments[experiment.experiment_id] = experiment

                return {
                    "status": "success",
                    "experiment_id": experiment.experiment_id,
                    "start_time": experiment.start_time.isoformat(),
                }
            except Exception as e:
                logger.error(f"Error creating A/B test: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/ab-test/{experiment_id}/results")
        async def get_ab_test_results(experiment_id: str):
            try:
                experiment = self.active_experiments.get(experiment_id)
                if not experiment:
                    raise HTTPException(status_code=404, detail="Experiment not found")

                results = self._calculate_experiment_results(experiment)
                return {
                    "status": "success",
                    "results": results,
                    "is_active": experiment.is_active,
                }
            except Exception as e:
                logger.error(f"Error getting A/B test results: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    def _log_version_to_mlflow(self, version: ModelVersion):
        """Log model version to MLflow."""
        try:
            with mlflow.start_run(run_name=f"version_{version.version_id}"):
                # Log parameters
                mlflow.log_params(version.metadata)

                # Log model
                mlflow.log_artifacts(version.model_path)

                # Log metrics
                mlflow.log_metrics(version.performance_metrics)

        except Exception as e:
            logger.error(f"Error logging to MLflow: {str(e)}")
            raise

    def _calculate_experiment_results(self, experiment: ABTest) -> Dict[str, Any]:
        """Calculate A/B test results with statistical significance."""
        results = {
            "model_a": experiment.results["model_a"],
            "model_b": experiment.results["model_b"],
            "statistical_significance": {},
        }

        # Calculate success rates
        success_rate_a = (
            experiment.results["model_a"]["successes"]
            / experiment.results["model_a"]["requests"]
            if experiment.results["model_a"]["requests"] > 0
            else 0
        )
        success_rate_b = (
            experiment.results["model_b"]["successes"]
            / experiment.results["model_b"]["requests"]
            if experiment.results["model_b"]["requests"] > 0
            else 0
        )

        # Perform statistical test
        if (
            experiment.results["model_a"]["requests"] > 0
            and experiment.results["model_b"]["requests"] > 0
        ):
            chi2, p_value = stats.chi2_contingency(
                [
                    [
                        experiment.results["model_a"]["successes"],
                        experiment.results["model_a"]["requests"]
                        - experiment.results["model_a"]["successes"],
                    ],
                    [
                        experiment.results["model_b"]["successes"],
                        experiment.results["model_b"]["requests"]
                        - experiment.results["model_b"]["successes"],
                    ],
                ]
            )[:2]

            results["statistical_significance"] = {
                "chi2": chi2,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }

        return results

    def update_experiment_metrics(
        self,
        experiment_id: str,
        model_version: str,
        success: bool,
        metrics: Dict[str, float],
    ):
        """Update experiment metrics."""
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            return

        if model_version == experiment.model_a.version_id:
            target = experiment.results["model_a"]
        elif model_version == experiment.model_b.version_id:
            target = experiment.results["model_b"]
        else:
            return

        target["requests"] += 1
        if success:
            target["successes"] += 1

        # Update metrics
        for metric_name, value in metrics.items():
            if metric_name not in target["metrics"]:
                target["metrics"][metric_name] = []
            target["metrics"][metric_name].append(value)

            # Update Prometheus metrics
            MODEL_PERFORMANCE.labels(model_version, metric_name).set(value)


def main():
    # Load configuration
    config = {
        "redis_host": "localhost",
        "redis_port": 6379,
        "mlflow_tracking_uri": "http://localhost:5000",
        "deployment_config": {"num_replicas": 2, "max_concurrent_queries": 100},
    }

    # Initialize service
    service = ModelVersioningService(config)

    # Start Ray Serve
    serve.start(http_options=HTTPOptions(host="0.0.0.0", port=8004))

    # Deploy application
    serve.run(
        service.app,
        name="sentient-avatar-versioning",
        route_prefix="/versioning",
        **config["deployment_config"],
    )


if __name__ == "__main__":
    main()
