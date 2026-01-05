# Training Approaches Comparison

This guide compares different ways to train a dialogue summarization model in this project.

## Approach 1: One-Line Training (Fastest to Start)

**Best for:** Quick experimentation, simple use cases

```python
from src.train import train_model_from_dataframes
from src.data_loader import DialogueDataLoader

loader = DialogueDataLoader()
train_df, _ = loader.load_train_test_data()

# Single function call
trainer = train_model_from_dataframes(
    train_df=train_df,
    model_name="facebook/bart-base",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-5
)
```

**Pros:**
- ✓ Minimal code (5 lines)
- ✓ Quick to start
- ✓ Good defaults

**Cons:**
- ✗ Less control over configuration
- ✗ Harder to customize pipeline

---

## Approach 2: Class-Based Training (Most Control)

**Best for:** Production, custom configurations, experimentation

```python
from src.data_loader import DialogueDataLoader
from src.train import DialogueSummarizationTrainer

# 1. Load data
loader = DialogueDataLoader(data_dir="data")
train_df, test_df = loader.load_train_test_data()

# 2. Initialize trainer with custom config
trainer = DialogueSummarizationTrainer(
    model_name="facebook/bart-base",
    output_dir="models/my-model",
    max_input_length=512,
    max_target_length=128,
    dialogue_format='standard'
)

# 3. Prepare data with custom preprocessing
dataset_dict = trainer.load_and_prepare_data(
    train_df=train_df,
    val_df=None,
    val_split=0.1,
    dialogue_col='dialogue',
    summary_col='summary'
)

# 4. Configure training with all options
training_args = trainer.get_training_arguments(
    learning_rate=2e-5,
    num_train_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    eval_steps=500,
    save_steps=500,
    logging_steps=100
)

# 5. Train with monitoring
trained_trainer = trainer.train(
    train_dataset=dataset_dict,
    training_args=training_args
)

# 6. Evaluate
eval_results = trained_trainer.evaluate()
```

**Pros:**
- ✓ Full control over all parameters
- ✓ Easy to customize each step
- ✓ Access to all trainer methods
- ✓ Can inspect intermediate results

**Cons:**
- ✗ More code to write
- ✗ Need to understand all parameters

---

## Approach 3: Command Line Scripts (Easiest)

**Best for:** Quick testing, beginners, standard workflows

### Quick Test (Subset)
```bash
python train_example.py
```

### Full Training
```bash
python src/train.py
```

**Pros:**
- ✓ No coding required
- ✓ Pre-configured settings
- ✓ Works out of the box

**Cons:**
- ✗ Limited customization (need to edit script)
- ✗ Harder to integrate into workflows

---

## Approach 4: Custom Trainer (Maximum Flexibility)

**Best for:** Research, special requirements, custom metrics

```python
from src.data_loader import DialogueDataLoader
from src.train import DialogueSummarizationTrainer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

# Initialize components
trainer_obj = DialogueSummarizationTrainer(
    model_name="facebook/bart-base",
    output_dir="models/custom"
)

# Load data
loader = DialogueDataLoader()
train_df, _ = loader.load_train_test_data()
dataset_dict = trainer_obj.load_and_prepare_data(train_df, val_split=0.1)

# Custom data collator with special handling
data_collator = DataCollatorForSeq2Seq(
    tokenizer=trainer_obj.tokenizer,
    model=trainer_obj.model,
    padding='max_length',
    max_length=512
)

# Custom metrics function
def custom_metrics(eval_pred):
    # Your custom evaluation logic
    result = trainer_obj.compute_metrics(eval_pred)
    # Add custom metrics
    result['custom_metric'] = your_custom_function(eval_pred)
    return result

# Custom training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="models/custom",
    # ... all custom parameters
)

# Create custom trainer
custom_trainer = Seq2SeqTrainer(
    model=trainer_obj.model,
    args=training_args,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['validation'],
    tokenizer=trainer_obj.tokenizer,
    data_collator=data_collator,
    compute_metrics=custom_metrics
)

# Train
custom_trainer.train()
```

**Pros:**
- ✓ Complete control
- ✓ Can customize everything
- ✓ Access to all HuggingFace features

**Cons:**
- ✗ Most complex
- ✗ Need deep understanding of HuggingFace

---

## Feature Comparison

| Feature | One-Line | Class-Based | Script | Custom |
|---------|----------|-------------|--------|--------|
| Lines of code | 5 | 20 | 0 | 30+ |
| Customization | Low | High | Medium | Maximum |
| Learning curve | Easy | Medium | Easy | Hard |
| Flexibility | Low | High | Low | Maximum |
| Best for | Quick tests | Production | Beginners | Research |

---

## Use Case Recommendations

### I want to quickly test if training works
→ **Use Command Line Script**
```bash
python train_example.py
```

### I'm experimenting with different models
→ **Use One-Line Training**
```python
for model in ["facebook/bart-base", "t5-small", "t5-base"]:
    train_model_from_dataframes(train_df, model_name=model, num_epochs=1)
```

### I'm building a production system
→ **Use Class-Based Training**
- Full control over configuration
- Easy to save/load configurations
- Can add custom validation

### I need custom metrics or loss functions
→ **Use Custom Trainer**
- Implement custom compute_metrics
- Override training_step if needed
- Full access to HuggingFace internals

### I'm tuning hyperparameters
→ **Use Class-Based with loops**
```python
for lr in [1e-5, 2e-5, 3e-5]:
    for batch_size in [4, 8]:
        trainer = DialogueSummarizationTrainer(...)
        training_args = trainer.get_training_arguments(
            learning_rate=lr, batch_size=batch_size
        )
        trainer.train(dataset_dict, training_args)
```

---

## Migration Path

### From Scripts → Class-Based

**Before (script):**
```bash
python train_example.py
```

**After (class-based):**
```python
# Copy the code from train_example.py
# Customize the parts you need
trainer = DialogueSummarizationTrainer(...)
# ... rest of code
```

### From One-Line → Class-Based

**Before:**
```python
train_model_from_dataframes(train_df, model_name="bart", num_epochs=3)
```

**After:**
```python
trainer = DialogueSummarizationTrainer(model_name="bart")
dataset = trainer.load_and_prepare_data(train_df)
args = trainer.get_training_arguments(num_train_epochs=3)
trainer.train(dataset, args)
```

### From Class-Based → Custom

**Add:**
```python
# Keep your existing code, then add:
from transformers import Seq2SeqTrainer

custom_trainer = Seq2SeqTrainer(
    model=trainer.model,  # Use model from DialogueSummarizationTrainer
    # ... custom configuration
)
```

---

## Performance Comparison

| Approach | Setup Time | Training Speed | Flexibility |
|----------|------------|----------------|-------------|
| One-Line | < 1 min | Same | ★☆☆ |
| Class-Based | 2-5 min | Same | ★★★ |
| Script | < 1 min | Same | ★☆☆ |
| Custom | 10+ min | Same | ★★★★★ |

Note: Training speed is the same - only setup complexity differs.

---

## Example Workflows

### Workflow 1: Research Experiment
```python
# 1. Quick test with script
!python train_example.py

# 2. If promising, use class-based for tuning
trainer = DialogueSummarizationTrainer(model_name="bart")
# ... tune parameters

# 3. Best model → custom trainer for paper experiments
custom_trainer = Seq2SeqTrainer(...)
# ... custom metrics for paper
```

### Workflow 2: Production Deployment
```python
# 1. Develop with class-based
trainer = DialogueSummarizationTrainer(...)
dataset = trainer.load_and_prepare_data(train_df)
trained = trainer.train(dataset)

# 2. Save configuration
import json
config = {
    'model_name': trainer.model_name,
    'max_input_length': trainer.max_input_length,
    # ... save all settings
}
with open('config.json', 'w') as f:
    json.dump(config, f)

# 3. Production: load config and train
with open('config.json') as f:
    config = json.load(f)
trainer = DialogueSummarizationTrainer(**config)
# ... train with saved config
```

---

## Summary

**Choose based on your needs:**

- **Learning/Testing** → Command line scripts
- **Simple training** → One-line function
- **Production** → Class-based approach
- **Research** → Custom trainer

**Most flexible choice:** Class-based approach
- Easy to start
- Can customize when needed
- Production-ready
- Well-documented

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed examples of each approach.
