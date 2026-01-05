"""
Evaluation Module

This module provides metrics and evaluation functions for dialogue summarization,
including ROUGE scores, BLEU, and other text generation metrics.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
    from nltk.translate.meteor_score import meteor_score
    import nltk
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: Some evaluation libraries not installed.")
    print("Install with: pip install rouge-score nltk")


class SummarizationEvaluator:
    """
    Evaluates dialogue summarization model performance using various metrics.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.rouge_scorer = None
        
        if METRICS_AVAILABLE:
            # Initialize ROUGE scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
                use_stemmer=True
            )
            
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                print("Downloading NLTK wordnet...")
                nltk.download('wordnet', quiet=True)
    
    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            
        Returns:
            Dictionary containing ROUGE scores
        """
        if not METRICS_AVAILABLE or self.rouge_scorer is None:
            raise ImportError("rouge-score package not installed")
        
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")
        
        scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': [],
            'rougeLsum': []
        }
        
        for pred, ref in zip(predictions, references):
            score = self.rouge_scorer.score(ref, pred)
            
            for metric in scores.keys():
                scores[metric].append(score[metric].fmeasure)
        
        # Compute averages
        avg_scores = {
            f"{metric}_precision": np.mean([
                self.rouge_scorer.score(ref, pred)[metric].precision
                for pred, ref in zip(predictions, references)
            ]),
            f"{metric}_recall": np.mean([
                self.rouge_scorer.score(ref, pred)[metric].recall
                for pred, ref in zip(predictions, references)
            ]),
            f"{metric}_fmeasure": np.mean(scores[metric])
            for metric in scores.keys()
        }
        
        return avg_scores
    
    def compute_bleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute BLEU scores.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            
        Returns:
            Dictionary containing BLEU scores
        """
        if not METRICS_AVAILABLE:
            raise ImportError("nltk package not installed")
        
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")
        
        # Tokenize
        pred_tokens = [pred.split() for pred in predictions]
        ref_tokens = [[ref.split()] for ref in references]
        
        # Compute sentence-level BLEU
        bleu_scores = [
            sentence_bleu(ref, pred)
            for pred, ref in zip(pred_tokens, ref_tokens)
        ]
        
        # Compute corpus-level BLEU
        corpus_bleu_score = corpus_bleu(ref_tokens, pred_tokens)
        
        return {
            'bleu_avg': np.mean(bleu_scores),
            'bleu_std': np.std(bleu_scores),
            'corpus_bleu': corpus_bleu_score
        }
    
    def compute_length_stats(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute length statistics.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            
        Returns:
            Dictionary containing length statistics
        """
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        return {
            'avg_pred_length': np.mean(pred_lengths),
            'avg_ref_length': np.mean(ref_lengths),
            'pred_length_std': np.std(pred_lengths),
            'ref_length_std': np.std(ref_lengths),
            'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths)
        }
    
    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            metrics: List of metrics to compute (default: all)
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        if metrics is None:
            metrics = ['rouge', 'bleu', 'length']
        
        results = {}
        
        if 'rouge' in metrics:
            try:
                results.update(self.compute_rouge(predictions, references))
            except Exception as e:
                print(f"Error computing ROUGE: {e}")
        
        if 'bleu' in metrics:
            try:
                results.update(self.compute_bleu(predictions, references))
            except Exception as e:
                print(f"Error computing BLEU: {e}")
        
        if 'length' in metrics:
            results.update(self.compute_length_stats(predictions, references))
        
        return results
    
    def print_evaluation_report(
        self,
        results: Dict[str, float],
        save_path: Optional[str] = None
    ):
        """
        Print formatted evaluation report.
        
        Args:
            results: Dictionary of evaluation results
            save_path: Optional path to save report
        """
        report = "\n" + "="*60 + "\n"
        report += "EVALUATION REPORT\n"
        report += "="*60 + "\n\n"
        
        # ROUGE scores
        report += "ROUGE Scores:\n"
        report += "-" * 40 + "\n"
        for key, value in results.items():
            if key.startswith('rouge'):
                report += f"  {key:20s}: {value:.4f}\n"
        
        # BLEU scores
        report += "\nBLEU Scores:\n"
        report += "-" * 40 + "\n"
        for key, value in results.items():
            if 'bleu' in key.lower():
                report += f"  {key:20s}: {value:.4f}\n"
        
        # Length statistics
        report += "\nLength Statistics:\n"
        report += "-" * 40 + "\n"
        for key, value in results.items():
            if 'length' in key or 'ratio' in key:
                report += f"  {key:20s}: {value:.4f}\n"
        
        report += "\n" + "="*60 + "\n"
        
        print(report)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")


def evaluate_model(
    predictions_file: str,
    references_file: str,
    output_file: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate model predictions from files.
    
    Args:
        predictions_file: Path to file with predicted summaries
        references_file: Path to file with reference summaries
        output_file: Optional path to save evaluation report
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load predictions and references
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f]
    
    with open(references_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f]
    
    # Evaluate
    evaluator = SummarizationEvaluator()
    results = evaluator.evaluate(predictions, references)
    
    # Print and save report
    evaluator.print_evaluation_report(results, save_path=output_file)
    
    return results


if __name__ == "__main__":
    # Example usage
    sample_predictions = [
        "Person A and B discussed hiking.",
        "They talked about weekend plans and outdoor activities."
    ]
    
    sample_references = [
        "Person A and Person B talked about hiking in the mountains.",
        "They discussed their weekend and outdoor activities."
    ]
    
    evaluator = SummarizationEvaluator()
    results = evaluator.evaluate(sample_predictions, sample_references)
    evaluator.print_evaluation_report(results)
