# Dialogue Summarization Project

A clean, scalable Python project for dialogue summarization using state-of-the-art NLP models.

## Project Overview

This project provides a complete pipeline for training, evaluating, and deploying dialogue summarization models. It leverages transformer-based models (e.g., BART, T5) to generate concise summaries of conversations.

## Project Structure

```
dialogue-summarization/
├── data/                          # Dataset directory
│   ├── train.csv                  # Training data with dialogues and summaries
│   ├── test.csv                   # Test data with dialogues only
│   └── sample_submission.csv      # Sample submission format
├── src/                           # Source code directory
│   ├── data_loader.py            # Data loading and batching utilities
│   ├── preprocess.py             # Text preprocessing functions
│   ├── train.py                  # Model training script
│   ├── inference.py              # Inference and prediction script
│   └── evaluate.py               # Evaluation metrics and reporting
├── models/                        # Directory for saved model checkpoints
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore file
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd dialogue-summarization
```

2. Install required dependencies:
```bash
pip install -r ../requirements.txt
```

## Usage

### Data Preparation

Place your training data in `data/train.csv` with the following format:
```csv
dialogue_id,dialogue,summary
1,"Person A: Hello! Person B: Hi there!","Greeting exchange"
```

### Training

Train a dialogue summarization model:
```bash
python src/train.py
```

Configure training parameters by editing the config dictionary in `train.py`.

### Inference

Generate summaries for new dialogues:
```bash
python src/inference.py
```

### Evaluation

Evaluate model performance using ROUGE metrics:
```bash
python src/evaluate.py
```

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Scalable Design**: Built to handle large datasets efficiently
- **Pre-trained Models**: Leverage Hugging Face transformers
- **Comprehensive Evaluation**: ROUGE score calculation and model comparison
- **Easy Configuration**: Simple configuration through dictionaries

## Model Architecture

The project uses transformer-based sequence-to-sequence models such as:
- BART (Bidirectional and Auto-Regressive Transformers)
- T5 (Text-to-Text Transfer Transformer)
- PEGASUS

## Evaluation Metrics

- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap  
- **ROUGE-L**: Longest common subsequence

## Future Enhancements

- Add support for additional evaluation metrics (BLEU, METEOR)
- Implement multi-GPU training
- Add hyperparameter tuning utilities
- Create web API for real-time inference
- Add data augmentation techniques

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please open an issue in the repository.
