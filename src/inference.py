"""
Inference Module

This module handles inference and prediction for dialogue summarization,
including loading trained models and generating summaries.
"""

from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig
)

from preprocess import DialoguePreprocessor


class DialogueSummarizer:
    """
    Handles inference for dialogue summarization using trained models.
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_input_length: int = 512,
        max_target_length: int = 128
    ):
        """
        Initialize the summarizer.
        
        Args:
            model_path: Path to trained model or Hugging Face model name
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            max_input_length: Maximum input sequence length
            max_target_length: Maximum summary length
        """
        self.model_path = Path(model_path) if "/" not in model_path else model_path
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize preprocessor
        self.preprocessor = DialoguePreprocessor(
            self.tokenizer,
            max_input_length=max_input_length,
            max_target_length=max_target_length
        )
        
        print(f"Model loaded successfully on {self.device}")
    
    def summarize(
        self,
        dialogue: Union[str, List[str]],
        num_beams: int = 4,
        length_penalty: float = 1.0,
        min_length: int = 10,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = False,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate summary for dialogue(s).
        
        Args:
            dialogue: Single dialogue string or list of dialogues
            num_beams: Number of beams for beam search
            length_penalty: Length penalty for beam search
            min_length: Minimum length of generated summary
            max_length: Maximum length of generated summary
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling instead of greedy decoding
            **kwargs: Additional generation parameters
            
        Returns:
            Generated summary or list of summaries
        """
        # Handle single string or list
        is_single = isinstance(dialogue, str)
        dialogues = [dialogue] if is_single else dialogue
        
        # Preprocess inputs
        inputs = self.tokenizer(
            dialogues,
            max_length=self.max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Set max_length if not provided
        if max_length is None:
            max_length = self.max_target_length
        
        # Generate summaries
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                num_beams=num_beams,
                length_penalty=length_penalty,
                min_length=min_length,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                early_stopping=True,
                **kwargs
            )
        
        # Decode summaries
        summaries = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Return single string or list based on input
        return summaries[0] if is_single else summaries
    
    def summarize_batch(
        self,
        dialogues: List[str],
        batch_size: int = 8,
        **generation_kwargs
    ) -> List[str]:
        """
        Generate summaries for a large batch of dialogues.
        
        Args:
            dialogues: List of dialogue strings
            batch_size: Batch size for processing
            **generation_kwargs: Arguments passed to summarize()
            
        Returns:
            List of generated summaries
        """
        all_summaries = []
        
        for i in range(0, len(dialogues), batch_size):
            batch = dialogues[i:i + batch_size]
            summaries = self.summarize(batch, **generation_kwargs)
            all_summaries.extend(summaries if isinstance(summaries, list) else [summaries])
            
            if (i + batch_size) % 100 == 0:
                print(f"Processed {min(i + batch_size, len(dialogues))}/{len(dialogues)} dialogues")
        
        return all_summaries
    
    def get_generation_config(
        self,
        strategy: str = "beam_search"
    ) -> GenerationConfig:
        """
        Get predefined generation configurations.
        
        Args:
            strategy: Generation strategy ('beam_search', 'sampling', 'diverse')
            
        Returns:
            GenerationConfig object
        """
        base_config = {
            "max_length": self.max_target_length,
            "min_length": 10,
            "early_stopping": True
        }
        
        if strategy == "beam_search":
            config = GenerationConfig(
                **base_config,
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=3
            )
        elif strategy == "sampling":
            config = GenerationConfig(
                **base_config,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )
        elif strategy == "diverse":
            config = GenerationConfig(
                **base_config,
                num_beams=5,
                num_beam_groups=5,
                diversity_penalty=1.0
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return config


def predict_on_test_set(
    model_path: str,
    test_dialogues: List[str],
    output_file: str,
    batch_size: int = 8
) -> List[str]:
    """
    Generate predictions for a test set and save to file.
    
    Args:
        model_path: Path to trained model
        test_dialogues: List of test dialogues
        output_file: Path to save predictions
        batch_size: Batch size for inference
        
    Returns:
        List of generated summaries
    """
    # Initialize summarizer
    summarizer = DialogueSummarizer(model_path)
    
    # Generate summaries
    print(f"Generating summaries for {len(test_dialogues)} dialogues...")
    summaries = summarizer.summarize_batch(
        test_dialogues,
        batch_size=batch_size,
        num_beams=4,
        length_penalty=1.0
    )
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for summary in summaries:
            f.write(summary + '\n')
    
    print(f"Predictions saved to {output_file}")
    
    return summaries


if __name__ == "__main__":
    # Example usage
    MODEL_PATH = "models/dialogue-summarization/final_model"
    
    # Example dialogue
    sample_dialogue = """
    Person A: Hey, how was your weekend?
    Person B: It was great! I went hiking with some friends.
    Person A: That sounds fun. Where did you go?
    Person B: We went to the mountains up north. The views were amazing.
    """
    
    # Initialize summarizer
    summarizer = DialogueSummarizer(MODEL_PATH)
    
    # Generate summary
    summary = summarizer.summarize(sample_dialogue)
    print(f"\nOriginal Dialogue:\n{sample_dialogue}")
    print(f"\nGenerated Summary:\n{summary}")
