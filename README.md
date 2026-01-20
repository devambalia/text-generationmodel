# LSTM Text Generator

A character-level LSTM text generator trained on Shakespeare's works using TensorFlow/Keras.

## Project Overview

This project implements an LSTM-based neural network to generate text in the style of Shakespeare. The model uses character-level tokenization and learns patterns from the training data to generate coherent text sequences.

## Features

- Character-level text generation
- LSTM architecture with embedding layer
- Temperature-controlled sampling for diverse outputs
- Model checkpointing and validation

## Requirements

- Python 3.7+
- TensorFlow 2.10.0+
- NumPy 1.21.0+

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The model is trained on Shakespeare's complete works. The dataset file `shakespeare.txt` should be placed in the project directory.

**Dataset Source:** 
- Download from: https://www.gutenberg.org/ebooks/100
- Or use any Shakespeare text corpus

## Usage

### Training the Model

```bash
python lstm_text_generator.py
```

The script will:
1. Load and preprocess the text data
2. Create character-level sequences
3. Train the LSTM model
4. Save the trained model as `shakespeare_lstm.h5`
5. Generate sample text outputs

### Model Architecture

- **Embedding Layer**: 64 dimensions
- **LSTM Layers**: Two LSTM layers with 128 units each
- **Dense Layer**: Softmax activation for character prediction
- **Sequence Length**: 30 characters
- **Training**: 10 epochs with validation split

### Generated Output

Sample outputs are saved to `generated_output.txt` after training. The model generates text based on seed phrases like:
- "to be or not to be"
- "romeo romeo wherefore"
- "shall i compare thee"

## Files

- `lstm_text_generator.py` - Main training and generation script
- `requirements.txt` - Python dependencies
- `shakespeare_lstm.h5` - Trained model (saved after training)
- `generated_output.txt` - Sample generated text outputs
- `README.md` - This file

## Model Performance

- Final Training Loss: ~1.41
- Final Training Accuracy: ~55.4%
- Validation Loss: ~1.84
- Validation Accuracy: ~45.8%

## Notes

- The model uses character-level tokenization, which allows it to learn character patterns and generate text character by character
- Temperature sampling (0.7) is used to control the randomness of generated text
- The model is trained on 100,000 characters from the Shakespeare dataset
- Training time: ~10-15 minutes on CPU

## Author

LSTM Text Generator Project
