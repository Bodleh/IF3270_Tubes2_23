import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Embedding, LSTM, Bidirectional, Dropout, Dense
import json
from data_loader import prepare_data
from sklearn.metrics import f1_score
import numpy as np
import os

def build_model(num_classes, vocab_size, emb_dim, seq_len, lstm_layers_config, dropout_rate=0.5):
    model = Sequential()
    model.add(tf.keras.Input(shape=(seq_len,), dtype=tf.int64))
    model.add(Embedding(input_dim=vocab_size, output_dim=emb_dim))

    for i, config in enumerate(lstm_layers_config):
        is_last_lstm = (i == len(lstm_layers_config) - 1)
        layer_type, units = config
        return_sequences = not is_last_lstm
        
        if layer_type == 'unidirectional':
            model.add(LSTM(units, return_sequences=return_sequences))
        elif layer_type == 'bidirectional':
            model.add(Bidirectional(LSTM(units, return_sequences=return_sequences)))
    
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=num_classes, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_evaluate(model_name, model_config):
    """A helper function to train and evaluate a model, then save its history and weights."""
    print(f"\nTraining model: {model_name}")
    
    model = build_model(
        num_classes=NUM_CLASSES,
        vocab_size=MAX_VOCAB_SIZE,
        emb_dim=EMBEDDING_DIM,
        seq_len=MAX_SEQUENCE_LENGTH,
        lstm_layers_config=model_config
    )
    model.summary()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    with open(f'results/{model_name}_history.json', 'w') as f:
        json.dump(history.history, f)
    model.save_weights(f'models/{model_name}.weights.h5')
    
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Macro F1-Score for {model_name}: {macro_f1:.4f}")
    
    return {
        'name': model_name,
        'history': history.history,
        'f1_score': macro_f1
    }

def run_experiments():
    results = {}

    print("\n\nExperiment 1: Number of LSTM Layers")
    num_layers_configs = {
        '1_layer_lstm': [('bidirectional', 64)],
        '2_layers_lstm': [('bidirectional', 64), ('bidirectional', 32)],
        '3_layers_lstm': [('bidirectional', 64), ('bidirectional', 32), ('bidirectional', 16)],
    }
    results['num_layers'] = [train_and_evaluate(name, config) for name, config in num_layers_configs.items()]

    print("\n\nExperiment 2: LSTM Cell Count")
    cell_count_configs = {
        'lstm_32_cells': [('bidirectional', 32)],
        'lstm_64_cells': [('bidirectional', 64)],
        'lstm_128_cells': [('bidirectional', 128)],
    }
    results['cell_count'] = [train_and_evaluate(name, config) for name, config in cell_count_configs.items()]

    print("\n\nExperiment 3: LSTM Layer Directionality")
    direction_configs = {
        'unidirectional_lstm': [('unidirectional', 64)],
        'bidirectional_lstm': [('bidirectional', 64)],
    }
    results['direction'] = [train_and_evaluate(name, config) for name, config in direction_configs.items()]
    
    with open('results/all_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nAll experiments complete. Results saved to 'all_experiment_results.json'")
    return results

if __name__ == '__main__':
    
    MAX_VOCAB_SIZE = 15000
    MAX_SEQUENCE_LENGTH = 250
    EMBEDDING_DIM = 128
    EPOCHS = 10
    BATCH_SIZE = 64

    train_ds, val_ds, test_ds, vectorize_layer, NUM_CLASSES, _ = prepare_data(
        max_vocab_size=MAX_VOCAB_SIZE, 
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE
    )
    
    run_experiments()