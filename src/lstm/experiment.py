import json
from data_loader import prepare_data
from lstm_keras import build_keras_lstm_model, train_model, evaluate_model

def build_train_evaluate(train_ds, val_ds, test_ds, epochs, model_name, num_classes, vocab_size, emb_dim, seq_len, lstm_layer_config):
    model = build_keras_lstm_model(
        num_classes=num_classes,
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        seq_len=seq_len,
        lstm_layers_config=lstm_layer_config
    )
    
    train_result = train_model(model, train_ds, val_ds, epochs, model_name=model_name)
    f1_score = evaluate_model(model, test_ds)
    
    return {
        'name': model_name,
        'history': train_result['history'],
        'f1_score': f1_score
    }

def run_experiments():
    MAX_VOCAB_SIZE = 15000
    MAX_SEQUENCE_LENGTH = 250
    EMBEDDING_DIM = 128
    EPOCHS = 20
    BATCH_SIZE = 64
    
    train_ds, val_ds, test_ds, _, NUM_CLASSES, _ = prepare_data(
        max_vocab_size=MAX_VOCAB_SIZE, 
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE
    )
    
    results = {}

    print("\n\nExperiment 1: Number of LSTM Layers")
    num_layers_configs = {
        '1_layer_lstm': [('bidirectional', 64)],
        '2_layers_lstm': [('bidirectional', 64), ('bidirectional', 32)],
        '3_layers_lstm': [('bidirectional', 64), ('bidirectional', 32), ('bidirectional', 16)],
    }
    results['num_layers'] = [build_train_evaluate(train_ds, val_ds, test_ds, EPOCHS, name, NUM_CLASSES, MAX_VOCAB_SIZE, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, config) for name, config in num_layers_configs.items()]

    print("\n\nExperiment 2: LSTM Cell Count")
    cell_count_configs = {
        'lstm_32_cells': [('unidirectional', 32)],
        'lstm_64_cells': [('unidirectional', 64)],
        'lstm_128_cells': [('unidirectional', 128)],
    }
    results['cell_count'] = [build_train_evaluate(train_ds, val_ds, test_ds, EPOCHS, name, NUM_CLASSES, MAX_VOCAB_SIZE, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, config) for name, config in cell_count_configs.items()]

    print("\n\nExperiment 3: LSTM Layer Directionality")
    direction_configs = {
        'unidirectional_lstm': [('unidirectional', 64)],
        'bidirectional_lstm': [('bidirectional', 64)],
    }
    results['direction'] = [build_train_evaluate(train_ds, val_ds, test_ds, EPOCHS, name, NUM_CLASSES, MAX_VOCAB_SIZE, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, config) for name, config in direction_configs.items()]
    
    with open('results/all_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nAll experiments complete. Results saved to 'all_experiment_results.json'")
    return results

if __name__ == '__main__':
    run_experiments()