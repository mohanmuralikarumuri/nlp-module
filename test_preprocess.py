"""
Test preprocessing with real dialogue data
"""

import sys
sys.path.append('src')

from data_loader import DialogueDataLoader
from preprocess import DialoguePreprocessor, preprocess_dialogue_data

print("="*60)
print("Testing Preprocessing with Real Dialogue Data")
print("="*60)

# Load real data
print("\n1. Loading data...")
loader = DialogueDataLoader(data_dir="data")
train_df, test_df = loader.load_train_test_data()

print(f"Loaded {len(train_df)} training samples")
print(f"Loaded {len(test_df)} test samples")

# Show sample raw dialogue
print("\n2. Sample Raw Dialogue:")
print("-"*60)
sample = train_df.iloc[0]
print(f"ID: {sample['id']}")
print(f"Dialogue:\n{sample['dialogue'][:300]}...")
print(f"\nSummary: {sample['summary']}")

# Test preprocessing
print("\n3. Testing Different Format Styles:")
print("-"*60)

# Standard format
print("\nStandard format:")
preprocessor_std = DialoguePreprocessor(dialogue_format='standard')
formatted_std = preprocessor_std.format_dialogue(sample['dialogue'])
print(formatted_std[:200] + "...")

# Turns format
print("\nTurns format:")
preprocessor_turns = DialoguePreprocessor(dialogue_format='turns')
formatted_turns = preprocessor_turns.format_dialogue(sample['dialogue'])
print(formatted_turns[:200] + "...")

# Compact format
print("\nCompact format:")
preprocessor_compact = DialoguePreprocessor(dialogue_format='compact')
formatted_compact = preprocessor_compact.format_dialogue(sample['dialogue'])
print(formatted_compact[:200] + "...")

# Preprocess subset of data
print("\n4. Preprocessing Data Subset:")
print("-"*60)

# Take a small subset for testing
subset_df = train_df.head(100).copy()
print(f"Processing {len(subset_df)} dialogues...")

preprocessor = DialoguePreprocessor(dialogue_format='standard')
processed_df = preprocessor.preprocess_dataframe(subset_df)

print(f"Processed {len(processed_df)} dialogues")

# Get statistics
print("\n5. Preprocessing Statistics:")
print("-"*60)
stats = preprocessor.get_dialogue_stats(processed_df)
print(f"Mean words per dialogue: {stats['dialogue']['mean_words']:.1f}")
print(f"Median words per dialogue: {stats['dialogue']['median_words']:.1f}")
print(f"Range: [{stats['dialogue']['min_words']}, {stats['dialogue']['max_words']}]")
print(f"Dialogues with speaker markers: {stats['speaker_info']['percentage_with_speakers']:.1f}%")

# Compare before and after
print("\n6. Before/After Comparison:")
print("-"*60)
comparison_idx = 2
print("BEFORE:")
print(subset_df.iloc[comparison_idx]['dialogue'][:200])
print("\nAFTER:")
print(processed_df.iloc[comparison_idx]['dialogue'][:200])

# Parse turns example
print("\n7. Parsing Dialogue Turns:")
print("-"*60)
turns = preprocessor.parse_dialogue_turns(subset_df.iloc[0]['dialogue'])
print(f"Found {len(turns)} turns in first dialogue:")
for i, (speaker, msg) in enumerate(turns[:5], 1):  # Show first 5 turns
    print(f"  {i}. {speaker}: {msg[:50]}...")

print("\n" + "="*60)
print("Preprocessing test completed successfully!")
print("="*60)
