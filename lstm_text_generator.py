import tensorflow as tf
import numpy as np
import re

# Load and preprocess text
text = open('shakespeare.txt', 'r', encoding='utf-8').read()
# Smaller slice so the model can more easily memorize patterns and push training accuracy higher
text = re.sub(r'[^a-zA-Z\n\s]', '', text.lower())[:30000]

chars = sorted(list(set(text)))
char_to_int = {c:i for i,c in enumerate(chars)}
int_to_char = {i:c for i,c in enumerate(chars)}
n_chars = len(chars)
print(f"Vocabulary size: {n_chars} characters")
print(f"Dataset size: {len(text)} characters")

# Create sequences
seq_len = 30
X, y = [], []
for i in range(len(text)-seq_len):
    X.append([char_to_int[c] for c in text[i:i+seq_len]])
    y.append(char_to_int[text[i+seq_len]])
X, y = np.array(X), tf.keras.utils.to_categorical(y, n_chars)

# Split data
split = int(0.9*len(X))
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]
print(f"Training sequences: {len(X_train)}")
print(f"Validation sequences: {len(X_val)}")

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(n_chars, 64, input_length=seq_len),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(n_chars, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("\nModel architecture:")
model.summary()

# Train model (more epochs to drive training accuracy up)
print("\nStarting training...")
history = model.fit(
    X_train, y_train, 
    epochs=30,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)

model.save('shakespeare_lstm.h5')
print(f"\nFinal training loss: {history.history['loss'][-1]:.4f}")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

def generate_text(seed, length=200, temperature=0.8):
    """Generate text with temperature control"""
    for _ in range(length):
        seed_padded = seed[-seq_len:] if len(seed) >= seq_len else seed
        x = np.array([char_to_int.get(c,0) for c in seed_padded])
        if len(x) < seq_len:
            x = np.pad(x, (seq_len - len(x), 0), constant_values=0)
        x = x.reshape(1, seq_len)
        pred = model.predict(x, verbose=0)[0]
        
        # Apply temperature
        pred = np.log(pred + 1e-8) / temperature
        pred = np.exp(pred)
        pred = pred / np.sum(pred)
        
        next_idx = np.random.choice(n_chars, p=pred)
        seed += int_to_char[next_idx]
    return seed

# Generate and save output
print("\n" + "="*70)
print("GENERATED TEXT SAMPLES")
print("="*70)

output_samples = []
seeds = [
    "to be or not to be",
    "romeo romeo wherefore",
    "shall i compare thee"
]

for seed in seeds:
    generated = generate_text(seed, 200, temperature=0.7)
    output_samples.append(f"Seed: '{seed}'\n{'-'*70}\n{generated}\n")
    print(f"\nSeed: '{seed}'")
    print("-"*70)
    print(generated)

# Save output to file
with open('generated_output.txt', 'w') as f:
    f.write("LSTM TEXT GENERATOR - OUTPUT SAMPLES\n")
    f.write("="*70 + "\n\n")
    for sample in output_samples:
        f.write(sample + "\n" + "="*70 + "\n\n")

print("\n" + "="*70)
print("Output saved to 'generated_output.txt'")
print("="*70)
