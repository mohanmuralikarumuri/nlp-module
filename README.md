# Dialogue Summarization System

A production-ready dialogue summarization system built with Python and Hugging Face Transformers. This project provides end-to-end tools for training, evaluating, and deploying dialogue summarization models.

## 🚀 Features

- **Scalable Training Pipeline**: Train state-of-the-art summarization models (BART, T5, PEGASUS)
- **Comprehensive Evaluation**: ROUGE, BLEU, and custom metrics
- **Interactive Web App**: Streamlit-based interface for real-time summarization
- **Clean Code Structure**: Modular design following ML best practices
- **Notebook Exploration**: Jupyter notebooks for data analysis and experimentation

## 📁 Project Structure

```
dialogue-summarization/
├── data/                          # Dataset storage
│   ├── train.csv                 # Training data
│   ├── test.csv                  # Test data
│   └── sample_submission.csv     # Submission template
│
├── notebooks/                     # Jupyter notebooks
│   └── exploration.ipynb         # Data exploration and analysis
│
├── src/                          # Source code
│   ├── data_loader.py           # Data loading utilities
│   ├── preprocess.py            # Text preprocessing and tokenization
│   ├── train.py                 # Training pipeline
│   ├── inference.py             # Model inference
│   └── evaluate.py              # Evaluation metrics
│
├── app/                          # Deployment application
│   ├── app.py                   # Streamlit web app
│   └── requirements.txt         # App dependencies
│
├── README.md                     # Project documentation
└── .gitignore                   # Git ignore rules
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/dialogue-summarization.git
   cd dialogue-summarization
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # Base dependencies
   pip install pandas numpy pathlib
   
   # For training (includes transformers, datasets, etc.)
   pip install -r requirements-training.txt
   
   # For web app deployment
   pip install -r app/requirements.txt
   ```

## 📊 Data Format

Your data should be in CSV format with the following columns:

- `id`: Unique identifier for each dialogue
- `dialogue`: The input conversation/dialogue text
- `summary`: The target summary (for training data only)

Example:
```csv
id,dialogue,summary
1,"Alice: Hello!\nBob: Hi there!","A greeting exchange"
2,"Alice: How are you?\nBob: I'm good!","Alice asks Bob how he is"
```

**Included Dataset:**
- Training: 10,311 dialogues with summaries
- Test: 2,210 dialogues (no summaries)

## 🎯 Quick Start

### 1. Data Loading and Exploration

Load and explore your dataset:

```python
from src.data_loader import DialogueDataLoader

# Load data
loader = DialogueDataLoader(data_dir="data")
train_df, test_df = loader.load_train_test_data()

# Get dataset statistics
stats = loader.get_dataset_info(train_df)
print(f"Training samples: {stats['num_samples']}")
print(f"Avg dialogue length: {stats['avg_dialogue_words']:.1f} words")
print(f"Avg summary length: {stats['avg_summary_words']:.1f} words")
```

Or use the exploration notebook:
```bash
jupyter notebook notebooks/exploration.ipynb
```

### 2. Training a Model

**Quick training (with example script):**

```bash
# Train on subset for testing
python train_example.py
```

**Full training pipeline:**

```python
from src.data_loader import DialogueDataLoader
from src.train import DialogueSummarizationTrainer

# Load data
loader = DialogueDataLoader(data_dir="data")
train_df, test_df = loader.load_train_test_data()

# Initialize trainer
trainer = DialogueSummarizationTrainer(
    model_name="facebook/bart-base",  # or "t5-small", "google/pegasus-cnn_dailymail"
    output_dir="models/my-model",
    max_input_length=512,
    max_target_length=128,
    dialogue_format='standard'  # 'standard', 'turns', or 'compact'
)

# Load and prepare data (includes preprocessing and tokenization)
dataset_dict = trainer.load_and_prepare_data(
    train_df=train_df,
    val_split=0.1  # 10% validation split
)

# Configure and train
training_args = trainer.get_training_arguments(
    learning_rate=2e-5,
    num_train_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=2
)

trained_trainer = trainer.train(dataset_dict, training_args)

# Evaluate
eval_results = trained_trainer.evaluate()
print(f"ROUGE-1: {eval_results['eval_rouge1']:.4f}")
```

**Convenience function:**

```python
from src.train import train_model_from_dataframes

# Train with single function call
trained_trainer = train_model_from_dataframes(
    train_df=train_df,
    model_name="facebook/bart-base",
    output_dir="models/my-model",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-5
)
```

**Monitor training:**
```bash
tensorboard --logdir models/my-model/logs
```

**For detailed training options, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md)**

### 3. Running Inference

```python
from src.inference import DialogueSummarizer

# Initialize summarizer
summarizer = DialogueSummarizer(
    model_path="models/dialogue-summarization/final_model"
)

# Generate summary
dialogue = "Person A: How was your day? Person B: It was great, thanks for asking!"
summary = summarizer.summarize(dialogue)
print(summary)
```

### 4. Evaluation

```python
from src.evaluate import SummarizationEvaluator

# Initialize evaluator
evaluator = SummarizationEvaluator()

# Evaluate predictions
results = evaluator.evaluate(
    predictions=predicted_summaries,
    references=reference_summaries
)

# Print report
evaluator.print_evaluation_report(results)
```

### 5. Launch Web App

```bash
streamlit run app/app.py
```

Then open your browser to `http://localhost:8501`

## � Documentation

Comprehensive guides are available:

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete training documentation
  - Supported models (BART, T5, PEGASUS)
  - Training pipeline and configuration
  - Hyperparameter tuning
  - Monitoring and troubleshooting
  - Advanced training techniques

- **[PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md)** - Text preprocessing
  - Dialogue format styles
  - Text cleaning and normalization
  - Speaker preservation
  - Tokenization strategies

- **[DATA_LOADER_GUIDE.md](DATA_LOADER_GUIDE.md)** - Data loading utilities
  - CSV loading and validation
  - Data cleaning and statistics
  - Train/validation splitting
  - HuggingFace dataset conversion

## 📈 Model Training Tips

### Recommended Models

| Model | Size | Best For | Learning Rate |
|-------|------|----------|---------------|
| `facebook/bart-base` | 139M | General use, fast training | 2e-5 |
| `facebook/bart-large` | 406M | Better quality, slower | 2e-5 |
| `t5-small` | 60M | Memory-constrained, fast | 1e-4 |
| `t5-base` | 220M | Good balance | 1e-4 |
| `google/pegasus-cnn_dailymail` | 568M | High-quality summaries | 5e-5 |

### Hyperparameter Guidelines

```python
# Quick experimentation (subset of data)
trainer = DialogueSummarizationTrainer(
    model_name="facebook/bart-base",
    dialogue_format='compact'
)
training_args = trainer.get_training_arguments(
    num_train_epochs=1,
    batch_size=8,
    eval_steps=100
)

# Production training (full dataset)
trainer = DialogueSummarizationTrainer(
    model_name="facebook/bart-large",
    max_input_length=1024,
    dialogue_format='standard'
)
training_args = trainer.get_training_arguments(
    learning_rate=2e-5,
    num_train_epochs=5,
    batch_size=2,
    gradient_accumulation_steps=8,
    generation_num_beams=6
)

# Memory-constrained environments
trainer = DialogueSummarizationTrainer(
    model_name="t5-small",
    max_input_length=256,
    dialogue_format='compact'
)
training_args = trainer.get_training_arguments(
    batch_size=1,
    gradient_accumulation_steps=16
)
```

**See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed configuration examples**

## 🧪 Evaluation Metrics

The system supports multiple evaluation metrics:

- **ROUGE-1, ROUGE-2, ROUGE-L**: Measures n-gram overlap
- **BLEU**: Machine translation quality metric
- **Length Statistics**: Analyzes compression ratios
- **Custom Metrics**: Easily extensible for domain-specific metrics

## 🚢 Deployment

### Option 1: Streamlit Cloud

1. Push your code to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click

### Option 2: Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install -r app/requirements.txt

CMD ["streamlit", "run", "app/app.py"]
```

Build and run:
```bash
docker build -t dialogue-summarization .
docker run -p 8501:8501 dialogue-summarization
```

## 📚 Documentation

### Module Overview

- **`data_loader.py`**: Handles data loading from various sources (CSV, HuggingFace datasets)
- **`preprocess.py`**: Text cleaning, tokenization, and dataset preparation
- **`train.py`**: Complete training pipeline with checkpoint management
- **`inference.py`**: Model loading and summary generation
- **`evaluate.py`**: Comprehensive evaluation metrics and reporting
- **`app.py`**: Interactive Streamlit web application

### Key Classes

- `DialogueDataLoader`: Data loading and format conversion
- `DialoguePreprocessor`: Text preprocessing and tokenization
- `DialogueSummarizationTrainer`: Model training and management
- `DialogueSummarizer`: Inference and prediction
- `SummarizationEvaluator`: Metric computation and evaluation

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the amazing NLP library
- [Streamlit](https://streamlit.io/) for the web app framework
- The open-source ML community for inspiration and tools

## 📧 Contact

For questions or support, please open an issue or contact [your-email@example.com]

## 🔗 Useful Resources

- [Hugging Face Model Hub](https://huggingface.co/models?pipeline_tag=summarization)
- [ROUGE Score Documentation](https://github.com/google-research/google-research/tree/master/rouge)
- [Seq2Seq Models Guide](https://huggingface.co/docs/transformers/tasks/summarization)

---

**Built with ❤️ for the NLP community**
