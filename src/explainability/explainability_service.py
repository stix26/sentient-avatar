import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lime
import lime.lime_text
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import ray
import seaborn as sns
import shap
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field
from ray import serve
from ray.serve.config import HTTPOptions
from ray.serve.deployment import Deployment
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
EXPLANATION_REQUESTS = Counter(
    "explanation_requests_total",
    "Total number of explanation requests",
    ["explanation_type"],
)
EXPLANATION_TIME = Histogram(
    "explanation_time_seconds",
    "Time taken to generate explanations",
    ["explanation_type"],
)
MODEL_CONFIDENCE = Gauge(
    "model_confidence_score",
    "Model confidence score",
    ["model_version", "prediction_type"],
)


class ModelExplainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.explanation_methods = {
            "shap": self._explain_with_shap,
            "lime": self._explain_with_lime,
            "attention": self._explain_with_attention,
            "integrated_gradients": self._explain_with_integrated_gradients,
        }
        self.visualization_methods = {
            "heatmap": self._create_heatmap,
            "bar_chart": self._create_bar_chart,
            "scatter_plot": self._create_scatter_plot,
        }

    def _explain_with_shap(self, model: nn.Module, input_text: str) -> Dict[str, Any]:
        """Generate SHAP explanations."""
        try:
            # Initialize SHAP explainer
            explainer = shap.Explainer(model)

            # Generate explanations
            shap_values = explainer([input_text])

            # Process results
            explanation = {
                "feature_importance": shap_values.values.tolist(),
                "base_value": shap_values.base_values.tolist(),
                "feature_names": shap_values.feature_names,
            }

            return explanation
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {str(e)}")
            raise

    def _explain_with_lime(self, model: nn.Module, input_text: str) -> Dict[str, Any]:
        """Generate LIME explanations."""
        try:
            # Initialize LIME explainer
            explainer = lime.lime_text.LimeTextExplainer(
                class_names=["negative", "positive"]
            )

            # Define prediction function
            def predict_proba(texts):
                return model(texts)

            # Generate explanations
            exp = explainer.explain_instance(input_text, predict_proba, num_features=10)

            # Process results
            explanation = {
                "feature_importance": exp.as_list(),
                "prediction": exp.predicted_label,
                "confidence": exp.score,
            }

            return explanation
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {str(e)}")
            raise

    def _explain_with_attention(
        self, model: nn.Module, input_text: str
    ) -> Dict[str, Any]:
        """Generate attention-based explanations."""
        try:
            # Get model attention weights
            attention_weights = model.get_attention_weights(input_text)

            # Process results
            explanation = {
                "attention_weights": attention_weights.tolist(),
                "token_importance": self._calculate_token_importance(attention_weights),
            }

            return explanation
        except Exception as e:
            logger.error(f"Error generating attention explanation: {str(e)}")
            raise

    def _explain_with_integrated_gradients(
        self, model: nn.Module, input_text: str, baseline: str = ""
    ) -> Dict[str, Any]:
        """Generate integrated gradients explanations."""
        try:
            # Initialize integrated gradients
            ig = IntegratedGradients(model)

            # Generate explanations
            attributions = ig.attribute(input_text, baseline=baseline, n_steps=50)

            # Process results
            explanation = {
                "attributions": attributions.tolist(),
                "feature_importance": self._calculate_feature_importance(attributions),
            }

            return explanation
        except Exception as e:
            logger.error(f"Error generating integrated gradients explanation: {str(e)}")
            raise

    def _calculate_token_importance(
        self, attention_weights: torch.Tensor
    ) -> List[float]:
        """Calculate token importance from attention weights."""
        # Implement token importance calculation
        return []

    def _calculate_feature_importance(self, attributions: torch.Tensor) -> List[float]:
        """Calculate feature importance from attributions."""
        # Implement feature importance calculation
        return []

    def _create_heatmap(self, data: np.ndarray, labels: List[str], title: str) -> str:
        """Create heatmap visualization."""
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(data, annot=True, fmt=".2f", cmap="YlOrRd")
            plt.title(title)
            plt.tight_layout()

            # Save plot
            plot_path = f"/tmp/heatmap_{int(time.time())}.png"
            plt.savefig(plot_path)
            plt.close()

            return plot_path
        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")
            raise

    def _create_bar_chart(
        self, data: List[float], labels: List[str], title: str
    ) -> str:
        """Create bar chart visualization."""
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(labels, data)
            plt.title(title)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save plot
            plot_path = f"/tmp/barchart_{int(time.time())}.png"
            plt.savefig(plot_path)
            plt.close()

            return plot_path
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            raise

    def _create_scatter_plot(
        self, x: List[float], y: List[float], labels: List[str], title: str
    ) -> str:
        """Create scatter plot visualization."""
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(x, y)
            for i, label in enumerate(labels):
                plt.annotate(label, (x[i], y[i]))
            plt.title(title)
            plt.tight_layout()

            # Save plot
            plot_path = f"/tmp/scatter_{int(time.time())}.png"
            plt.savefig(plot_path)
            plt.close()

            return plot_path
        except Exception as e:
            logger.error(f"Error creating scatter plot: {str(e)}")
            raise


class ExplainabilityService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.explainer = ModelExplainer(config)
        self.mlflow_client = MlflowClient()

        # Initialize FastAPI app
        self.app = FastAPI()
        self._setup_routes()

    def _setup_routes(self):
        """Set up FastAPI routes."""

        class ExplanationRequest(BaseModel):
            model_path: str
            input_text: str
            explanation_type: str
            visualization_type: Optional[str] = None

        @self.app.post("/explain")
        async def generate_explanation(request: ExplanationRequest):
            try:
                # Load model
                model = AutoModelForCausalLM.from_pretrained(request.model_path)
                model.to(self.device)

                # Generate explanation
                start_time = time.time()
                explanation = self.explainer.explanation_methods[
                    request.explanation_type
                ](model, request.input_text)

                # Update metrics
                EXPLANATION_TIME.labels(request.explanation_type).observe(
                    time.time() - start_time
                )
                EXPLANATION_REQUESTS.labels(request.explanation_type).inc()

                # Generate visualization if requested
                visualization_path = None
                if request.visualization_type:
                    visualization_path = self.explainer.visualization_methods[
                        request.visualization_type
                    ](
                        explanation["feature_importance"],
                        explanation.get("feature_names", []),
                        f"{request.explanation_type} Explanation",
                    )

                # Log to MLflow
                self._log_explanation(
                    request.model_path, request.explanation_type, explanation
                )

                return {
                    "status": "success",
                    "explanation": explanation,
                    "visualization_path": visualization_path,
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.error(f"Error generating explanation: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    def _log_explanation(
        self, model_path: str, explanation_type: str, explanation: Dict[str, Any]
    ):
        """Log explanation to MLflow."""
        try:
            with mlflow.start_run(run_name=f"explanation_{explanation_type}"):
                # Log parameters
                mlflow.log_params(
                    {"model_path": model_path, "explanation_type": explanation_type}
                )

                # Log metrics
                mlflow.log_metrics(
                    {
                        "explanation_time": EXPLANATION_TIME.labels(
                            explanation_type
                        ).observe()
                    }
                )

                # Log artifacts
                if "visualization_path" in explanation:
                    mlflow.log_artifact(explanation["visualization_path"])

        except Exception as e:
            logger.error(f"Error logging to MLflow: {str(e)}")
            raise


def main():
    # Load configuration
    config = {
        "mlflow_tracking_uri": "http://localhost:5000",
        "deployment_config": {"num_replicas": 2, "max_concurrent_queries": 100},
    }

    # Initialize service
    service = ExplainabilityService(config)

    # Start Ray Serve
    serve.start(http_options=HTTPOptions(host="0.0.0.0", port=8006))

    # Deploy application
    serve.run(
        service.app,
        name="sentient-avatar-explainability",
        route_prefix="/explain",
        **config["deployment_config"],
    )


if __name__ == "__main__":
    main()
