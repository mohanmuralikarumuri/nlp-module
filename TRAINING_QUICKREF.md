# Training Quick Reference

## One-Line Training

```python
from src.train import train_model_from_dataframes
from src.data_loader import DialogueDataLoader

loader = DialogueDataLoader()
train_df, _ = loader.load_train_test_data()

trained = train_model_from_dataframes(
    train_df, model_name="facebook/bart-base", num_epochs=3, batch_size=4
)
```

## Standard Training

```python
from src.train import DialogueSummarizationTrainer
from src.data_loader import DialogueDataLoader

# 1. Load data
loader = DialogueDataLoader()
train_df, _ = loader.load_train_test_data()

# 2. Init trainer
trainer = DialogueSummarizationTrainer(
    model_name="facebook/bart-base",
    output_dir="models/my-model"
)

# 3. Prepare data
dataset = trainer.load_and_prepare_data(train_df, val_split=0.1)

# 4. Train
trainer.train(dataset)
```

## Command Line

```bash
# Quick test (subset)
python train_example.py

# Full training
python src/train.py

# Monitor
tensorboard --logdir models/dialogue-summarization/logs
```

## Model Options

| Model | Command | Best For |
|-------|---------|----------|
| BART Base | `model_name="facebook/bart-base"` | Fast, general use |
| BART Large | `model_name="facebook/bart-large"` | Better quality |
| T5 Small | `model_name="t5-small"` | Memory-limited |
| T5 Base | `model_name="t5-base"` | Good balance |
| PEGASUS | `model_name="google/pegasus-cnn_dailymail"` | High quality |

## Common Configurations

### Fast Experimentation
```python
trainer = DialogueSummarizationTrainer(
    model_name="facebook/bart-base",
    dialogue_format='compact'
)
args = trainer.get_training_arguments(
    num_train_epochs=1, batch_size=8, eval_steps=100
)
```

### Production
```python
trainer = DialogueSummarizationTrainer(
    model_name="facebook/bart-large",
    max_input_length=1024
)
args = trainer.get_training_arguments(
    learning_rate=2e-5, num_train_epochs=5, 
    batch_size=2, gradient_accumulation_steps=8
)
```

### Memory-Constrained
```python
trainer = DialogueSummarizationTrainer(
    model_name="t5-small",
    max_input_length=256,
    dialogue_format='compact'
)
args = trainer.get_training_arguments(
    batch_size=1, gradient_accumulation_steps=16
)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of Memory | Reduce `batch_size`, increase `gradient_accumulation_steps` |
| Slow training | Reduce `eval_steps`, use smaller model, enable FP16 |
| Poor quality | Train longer, use larger model, try `dialogue_format='standard'` |
| Import errors | `pip install -r requirements-training.txt` |

## Key Parameters

```python
# Model configuration
model_name="facebook/bart-base"     # Model to use
max_input_length=512                # Max dialogue tokens
max_target_length=128               # Max summary tokens
dialogue_format='standard'          # 'standard', 'turns', 'compact'

# Training configuration
learning_rate=2e-5                  # 2e-5 for BART, 1e-4 for T5
num_train_epochs=3                  # Number of epochs
batch_size=4                        # Batch size per device
gradient_accumulation_steps=2       # Effective batch = batch_size * this
warmup_steps=500                    # LR warmup steps
eval_steps=500                      # Evaluate every N steps
generation_num_beams=4              # Beam search size
```

## Monitoring

```bash
# TensorBoard
tensorboard --logdir models/my-model/logs

# Check metrics
# - train/loss (should decrease)
# - eval/loss (should decrease)
# - eval/rouge1 (should increase)
```

## After Training

```python
# Load trained model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("models/my-model/final_model")
model = AutoModelForSeq2SeqLM.from_pretrained("models/my-model/final_model")

# Generate summary
inputs = tokenizer("Alice: Hi!\nBob: Hello!", return_tensors='pt', max_length=512)
outputs = model.generate(**inputs, max_length=128, num_beams=4)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Documentation

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete training documentation
- **[README.md](README.md)** - Project overview
- **[PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md)** - Preprocessing docs
