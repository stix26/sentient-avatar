import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from datasets import load_dataset, load_metric
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
from rouge_score import rouge_scorer
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from sacrebleu.metrics import BLEU, CHRF, TER
import nltk
nltk.download('punkt')

class ModelEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        self.bleu = BLEU()
        self.chrf = CHRF()
        self.ter = TER()
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_path"],
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_path"],
            padding_side="right",
            use_fast=True
        )
        
        # Initialize generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

    def evaluate_benchmarks(self) -> Dict[str, float]:
        """Evaluate model on standard benchmarks."""
        results = {}
        
        # Load benchmark datasets
        benchmarks = {
            "mmlu": load_dataset("cais/mmlu", "all"),
            "hellaswag": load_dataset("hellaswag"),
            "truthfulqa": load_dataset("truthful_qa", "generation"),
            "gsm8k": load_dataset("gsm8k", "main")
        }
        
        for name, dataset in benchmarks.items():
            if name == "mmlu":
                results[name] = self._evaluate_mmlu(dataset)
            elif name == "hellaswag":
                results[name] = self._evaluate_hellaswag(dataset)
            elif name == "truthfulqa":
                results[name] = self._evaluate_truthfulqa(dataset)
            elif name == "gsm8k":
                results[name] = self._evaluate_gsm8k(dataset)
        
        return results

    def evaluate_conversation_quality(self, conversations: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate conversation quality using multiple metrics."""
        results = {
            "coherence": [],
            "relevance": [],
            "fluency": [],
            "diversity": []
        }
        
        for conv in conversations:
            # Generate response
            response = self.generator(
                conv["context"],
                max_length=200,
                num_return_sequences=1,
                temperature=0.7
            )[0]["generated_text"]
            
            # Calculate metrics
            results["coherence"].append(self._calculate_coherence(conv["context"], response))
            results["relevance"].append(self._calculate_relevance(conv["context"], response))
            results["fluency"].append(self._calculate_fluency(response))
            results["diversity"].append(self._calculate_diversity(response))
        
        # Average metrics
        return {k: np.mean(v) for k, v in results.items()}

    def evaluate_safety(self, test_cases: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate model safety and alignment."""
        results = {
            "harmful_content": 0,
            "bias": 0,
            "toxicity": 0,
            "privacy": 0
        }
        
        for case in test_cases:
            response = self.generator(
                case["prompt"],
                max_length=200,
                num_return_sequences=1,
                temperature=0.7
            )[0]["generated_text"]
            
            # Check for harmful content
            results["harmful_content"] += self._check_harmful_content(response)
            
            # Check for bias
            results["bias"] += self._check_bias(response)
            
            # Check for toxicity
            results["toxicity"] += self._check_toxicity(response)
            
            # Check for privacy violations
            results["privacy"] += self._check_privacy(response)
        
        # Normalize results
        total = len(test_cases)
        return {k: v/total for k, v in results.items()}

    def _evaluate_mmlu(self, dataset) -> float:
        """Evaluate on MMLU benchmark."""
        correct = 0
        total = 0
        
        for example in dataset["test"]:
            prompt = f"Question: {example['question']}\nA) {example['A']}\nB) {example['B']}\nC) {example['C']}\nD) {example['D']}\nAnswer:"
            response = self.generator(prompt, max_length=50)[0]["generated_text"]
            
            # Extract answer
            answer = response.split("Answer:")[-1].strip()
            if answer in example["choices"]:
                correct += 1
            total += 1
        
        return correct / total

    def _evaluate_hellaswag(self, dataset) -> float:
        """Evaluate on HellaSwag benchmark."""
        correct = 0
        total = 0
        
        for example in dataset["validation"]:
            prompt = f"Context: {example['ctx']}\nA) {example['endings'][0]}\nB) {example['endings'][1]}\nC) {example['endings'][2]}\nD) {example['endings'][3]}\nAnswer:"
            response = self.generator(prompt, max_length=50)[0]["generated_text"]
            
            # Extract answer
            answer = response.split("Answer:")[-1].strip()
            if answer in example["endings"]:
                correct += 1
            total += 1
        
        return correct / total

    def _evaluate_truthfulqa(self, dataset) -> float:
        """Evaluate on TruthfulQA benchmark."""
        correct = 0
        total = 0
        
        for example in dataset["validation"]:
            prompt = f"Question: {example['question']}\nAnswer:"
            response = self.generator(prompt, max_length=200)[0]["generated_text"]
            
            # Check if response is truthful
            if self._check_truthfulness(response, example["correct_answers"]):
                correct += 1
            total += 1
        
        return correct / total

    def _evaluate_gsm8k(self, dataset) -> float:
        """Evaluate on GSM8K benchmark."""
        correct = 0
        total = 0
        
        for example in dataset["test"]:
            prompt = f"Question: {example['question']}\nLet's solve this step by step:"
            response = self.generator(prompt, max_length=200)[0]["generated_text"]
            
            # Extract answer
            try:
                answer = float(response.split("Answer:")[-1].strip())
                if abs(answer - example["answer"]) < 0.01:
                    correct += 1
            except:
                pass
            total += 1
        
        return correct / total

    def _calculate_coherence(self, context: str, response: str) -> float:
        """Calculate response coherence using BERTScore."""
        P, R, F1 = score([response], [context], lang="en", device=self.device)
        return F1.mean().item()

    def _calculate_relevance(self, context: str, response: str) -> float:
        """Calculate response relevance using ROUGE scores."""
        scores = self.rouge_scorer.score(context, response)
        return (scores["rouge1"].fmeasure + scores["rouge2"].fmeasure + scores["rougeL"].fmeasure) / 3

    def _calculate_fluency(self, text: str) -> float:
        """Calculate text fluency using language model perplexity."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            loss = outputs.loss
        return torch.exp(loss).item()

    def _calculate_diversity(self, text: str) -> float:
        """Calculate response diversity using vocabulary statistics."""
        words = text.split()
        unique_words = set(words)
        return len(unique_words) / len(words)

    def _check_harmful_content(self, text: str) -> float:
        """Check for harmful content using a safety classifier."""
        # Implement safety classifier
        return 0.0

    def _check_bias(self, text: str) -> float:
        """Check for bias using a bias classifier."""
        # Implement bias classifier
        return 0.0

    def _check_toxicity(self, text: str) -> float:
        """Check for toxicity using a toxicity classifier."""
        # Implement toxicity classifier
        return 0.0

    def _check_privacy(self, text: str) -> float:
        """Check for privacy violations using a privacy classifier."""
        # Implement privacy classifier
        return 0.0

    def _check_truthfulness(self, response: str, correct_answers: List[str]) -> bool:
        """Check if response is truthful by comparing with correct answers."""
        # Implement truthfulness checker
        return True

def main():
    # Load configuration
    config = {
        "model_path": "/app/models/trained",
        "benchmarks": ["mmlu", "hellaswag", "truthfulqa", "gsm8k"],
        "conversation_samples": 100,
        "safety_samples": 100
    }
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    
    # Run evaluations
    benchmark_results = evaluator.evaluate_benchmarks()
    conversation_results = evaluator.evaluate_conversation_quality([])
    safety_results = evaluator.evaluate_safety([])
    
    # Save results
    results = {
        "benchmarks": benchmark_results,
        "conversation_quality": conversation_results,
        "safety": safety_results,
        "timestamp": datetime.now().isoformat()
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 