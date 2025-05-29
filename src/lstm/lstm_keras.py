import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Embedding, LSTM, Bidirectional, Dropout, Dense
import json
from sklearn.metrics import f1_score
import numpy as np
import os

def build_keras_lstm_model(num_classes, vocab_size, emb_dim, seq_len, 
                              lstm_layers_config, 
                              dense_layers_config=None, 
                              dropout_rate=0.5):
    """
    Builds a Keras LSTM model for sequence classification with configurable dense layers.
    
    lstm_layers_config (list): Configuration for LSTM layers. 
                                Each element is a tuple (type, units),
                                e.g., [('unidirectional', 64), ('bidirectional', 32)].
                                'type' can be 'unidirectional' or 'bidirectional'.
                                
    dense_layers_config (list, optional): Configuration for intermediate dense layers
                                            before the final classification layer.
                                            Each element is an int specifying the number of units
                                            for a Dense layer with 'relu' activation.
                                            Defaults to None (no intermediate dense layers).
                                            Example: [128, 64] for two dense layers.
    """
    model = Sequential()
    
    model.add(tf.keras.Input(shape=(seq_len,), dtype=tf.int64))
    model.add(Embedding(input_dim=vocab_size, output_dim=emb_dim))

    if not lstm_layers_config:
        raise ValueError("lstm_layers_config cannot be empty. Please provide at least one LSTM layer configuration.")

    for i, config in enumerate(lstm_layers_config):
        is_last_lstm = (i == len(lstm_layers_config) - 1)
        
        if not isinstance(config, (list, tuple)) or len(config) != 2:
            raise ValueError(f"Each item in lstm_layers_config must be a tuple/list of (type, units). Got: {config}")
        
        layer_type, units = config
        
        if not isinstance(units, int) or units <= 0:
            raise ValueError(f"LSTM units must be a positive integer. Got: {units}")

        return_sequences = not is_last_lstm  

        if layer_type == 'unidirectional':
            model.add(LSTM(units, return_sequences=return_sequences, name=f'unidirectional_lstm_{i}'))
        elif layer_type == 'bidirectional':
            model.add(Bidirectional(LSTM(units, return_sequences=return_sequences)))
        else:
            raise ValueError(f"Invalid LSTM layer type: {layer_type}. Choose 'unidirectional' or 'bidirectional'.")

    model.add(Dropout(dropout_rate))

    if dense_layers_config:
        if not isinstance(dense_layers_config, list):
            raise ValueError("dense_layers_config must be a list of integers (units).")
        for i, dense_units in enumerate(dense_layers_config):
            if not isinstance(dense_units, int) or dense_units <= 0:
                raise ValueError(f"Dense layer units must be a positive integer. Got: {dense_units}")
            model.add(Dense(units=dense_units, activation='relu'))

    model.add(Dense(units=num_classes, activation='softmax', name='output_layer'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_ds, val_ds, epochs=10, model_name=None):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1
    )
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    if model_name == None:
        model_name = 'unk_lstm_model'
        
    with open(f'results/{model_name}_history.json', 'w') as f:
        json.dump(history.history, f)
    model.save_weights(f'models/{model_name}.weights.h5')
    
    return {
        'history': history.history,
        'model': model
    }

def evaluate_model(model, test_ds):
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Macro F1-Score: {macro_f1:.4f}")
    
    return macro_f1
    
    
