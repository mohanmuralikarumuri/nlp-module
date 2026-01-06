"""
Evaluation Module for Dialogue Summarization

This module provides evaluation metrics and utilities for assessing
the quality of generated summaries. Includes:
- ROUGE score calculation (ROUGE-1, ROUGE-2, ROUGE-L)
- BLEU score computation
- Model performance analysis and reporting
- Comparison utilities for different models
"""

from rouge_score import rouge_scorer
import pandas as pd


def calculate_rouge_scores(predictions, references):
    """
    Calculate ROUGE scores for generated summaries.
    
    Args:
        predictions (list): List of generated summaries
        references (list): List of reference (ground truth) summaries
        
    Returns:
        dict: Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': sum(rouge1_scores) / len(rouge1_scores),
        'rouge2': sum(rouge2_scores) / len(rouge2_scores),
        'rougeL': sum(rougeL_scores) / len(rougeL_scores)
    }


def evaluate_model(predictions_path, references_path):
    """
    Evaluate model predictions against reference summaries.
    
    Args:
        predictions_path (str): Path to CSV file with predicted summaries
        references_path (str): Path to CSV file with reference summaries
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Load predictions and references
    predictions_df = pd.read_csv(predictions_path)
    references_df = pd.read_csv(references_path)
    
    # Extract summaries
    predictions = predictions_df['summary'].tolist()
    references = references_df['summary'].tolist()
    
    # Calculate ROUGE scores
    rouge_scores = calculate_rouge_scores(predictions, references)
    
    return rouge_scores


def print_evaluation_report(scores):
    """
    Print a formatted evaluation report.
    
    Args:
        scores (dict): Dictionary containing evaluation metrics
    """
    print("\n" + "="*50)
    print("Dialogue Summarization - Evaluation Report")
    print("="*50)
    
    print(f"\nROUGE-1 F1 Score: {scores['rouge1']:.4f}")
    print(f"ROUGE-2 F1 Score: {scores['rouge2']:.4f}")
    print(f"ROUGE-L F1 Score: {scores['rougeL']:.4f}")
    
    print("\n" + "="*50)


def compare_models(predictions_paths, model_names, references_path):
    """
    Compare multiple models based on their predictions.
    
    Args:
        predictions_paths (list): List of paths to prediction CSV files
        model_names (list): List of model names corresponding to predictions
        references_path (str): Path to reference summaries CSV file
        
    Returns:
        pd.DataFrame: DataFrame comparing model performances
    """
    results = []
    
    for pred_path, model_name in zip(predictions_paths, model_names):
        scores = evaluate_model(pred_path, references_path)
        scores['model'] = model_name
        results.append(scores)
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    predictions_path = '../data/predictions.csv'
    references_path = '../data/validation.csv'  # Path to validation set with ground truth summaries
    
    scores = evaluate_model(predictions_path, references_path)
    print_evaluation_report(scores)
