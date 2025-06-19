import argparse
import logging
import os
from datetime import datetime
from typing import Any, Dict

import mlflow
import optuna
import ray
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistributedTrainingService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        # Initialize MLflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment("sentient-avatar-training")

        # Initialize Ray
        ray.init(address=os.getenv("RAY_ADDRESS", "auto"))

        # Initialize Weights & Biases
        wandb.init(
            project="sentient-avatar",
            config=self.config,
            name=f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )

    def prepare_model(self) -> tuple:
        """Prepare model for distributed training with LoRA and quantization."""
        logger.info("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"], load_in_8bit=True, device_map="auto"
        )

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config["lora_r"],
            lora_alpha=self.config["lora_alpha"],
            target_modules=self.config["lora_target_modules"],
            lora_dropout=self.config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA
        model = get_peft_model(model, lora_config)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"], padding_side="right", use_fast=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def prepare_dataset(self, tokenizer) -> Any:
        """Prepare and preprocess dataset for training."""
        logger.info("Loading and preprocessing dataset...")

        # Load dataset
        dataset = load_dataset(
            self.config["dataset_name"], split=self.config["dataset_split"]
        )

        # Tokenize dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config["max_length"],
            )

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=dataset.column_names
        )

        return tokenized_dataset

    def train(self):
        """Execute distributed training with hyperparameter optimization."""
        logger.info("Starting distributed training...")

        # Prepare model and dataset
        model, tokenizer = self.prepare_model()
        dataset = self.prepare_dataset(tokenizer)

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=self.config["output_dir"],
            num_train_epochs=self.config["num_epochs"],
            per_device_train_batch_size=self.config["batch_size"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            learning_rate=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            warmup_steps=self.config["warmup_steps"],
            logging_steps=self.config["logging_steps"],
            save_steps=self.config["save_steps"],
            evaluation_strategy="steps",
            eval_steps=self.config["eval_steps"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            deepspeed=self.config["deepspeed_config"],
            report_to=["wandb", "mlflow"],
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            ),
        )

        # Start hyperparameter optimization
        def objective(trial):
            # Sample hyperparameters
            config = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1e-3, log=True
                ),
                "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1),
                "warmup_steps": trial.suggest_int("warmup_steps", 100, 1000),
                "lora_r": trial.suggest_int("lora_r", 8, 32),
                "lora_alpha": trial.suggest_int("lora_alpha", 16, 64),
                "lora_dropout": trial.suggest_float("lora_dropout", 0.1, 0.3),
            }

            # Update trainer config
            trainer.args.learning_rate = config["learning_rate"]
            trainer.args.weight_decay = config["weight_decay"]
            trainer.args.warmup_steps = config["warmup_steps"]

            # Train and evaluate
            trainer.train()
            metrics = trainer.evaluate()

            return metrics["eval_loss"]

        # Create Optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.config["n_trials"])

        # Log best hyperparameters
        mlflow.log_params(study.best_params)
        wandb.log({"best_params": study.best_params})

        # Train with best hyperparameters
        logger.info("Training with best hyperparameters...")
        trainer.train()

        # Save final model
        trainer.save_model(self.config["output_dir"])
        tokenizer.save_pretrained(self.config["output_dir"])

        # Log final metrics
        final_metrics = trainer.evaluate()
        mlflow.log_metrics(final_metrics)
        wandb.log(final_metrics)

        logger.info("Training completed successfully!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.parse_args()

    # Load configuration
    config = {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "dataset_name": "sentient-avatar/conversations",
        "dataset_split": "train",
        "output_dir": "/app/models/trained",
        "num_epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "logging_steps": 100,
        "save_steps": 1000,
        "eval_steps": 500,
        "max_length": 2048,
        "n_trials": 10,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": ["q_proj", "v_proj"],
        "deepspeed_config": {
            "fp16": {"enabled": True},
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 2e-5, "weight_decay": 0.01},
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 2e-5,
                    "warmup_num_steps": 500,
                },
            },
            "gradient_clipping": 1.0,
            "train_batch_size": 16,
            "train_micro_batch_size_per_gpu": 4,
            "gradient_accumulation_steps": 4,
        },
    }

    # Start training service
    service = DistributedTrainingService(config)
    service.train()


if __name__ == "__main__":
    main()
