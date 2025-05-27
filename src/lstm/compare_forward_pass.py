import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score
from data_loader import prepare_data
from train import build_model
from forward_scratch import EmbeddingScratch, BidirectionalLSTMSratch, DropoutScratch, DenseScratch

def build_model_from_scratch(keras_model):
    scratch_layers = []
    
    for layer in keras_model.layers:
        layer_name = layer.name.lower()
        weights = layer.get_weights()
        
        if 'embedding' in layer_name:
            scratch_layers.append(EmbeddingScratch(weights))
        elif 'bidirectional' in layer_name:
            scratch_layers.append(BidirectionalLSTMSratch(weights))
        elif 'dropout' in layer_name:
            scratch_layers.append(DropoutScratch(layer.rate))
        elif 'dense' in layer_name:
            scratch_layers.append(DenseScratch(weights))
            
    return scratch_layers

def forward_pass_scratch(x_input_batch, scratch_model):
    output = x_input_batch
    for layer in scratch_model:
        if isinstance(layer, DropoutScratch):
             output = layer.forward(output, training=False)
        else:
            output = layer.forward(output)
    return output

def main():    
    BATCH_SIZE = 32
    
    _, _, _, _, NUM_CLASSES, (X_test, y_test) = prepare_data(
        max_vocab_size=15000, 
        max_sequence_length=250,
        batch_size=BATCH_SIZE
    )
    
    # Keras
    MODEL_NAME = 'bidirectional_lstm'
    MODEL_CONFIG = [('bidirectional', 64)]
    
    print(f"Loading Keras model '{MODEL_NAME}'...")
    keras_model = build_model(
        num_classes=NUM_CLASSES,
        vocab_size=15000,
        emb_dim=128,
        seq_len=250,
        lstm_layers_config=MODEL_CONFIG
    )
    try:
        keras_model.load_weights(f'models/{MODEL_NAME}.weights.h5')
    except FileNotFoundError:
        print(f"Weight file '{MODEL_NAME}.weights.h5' not found. Please run train.py first.")
        return

    print(f"Performing prediction with Keras (Batch Size: {BATCH_SIZE})...")
    keras_probs = keras_model.predict(X_test, batch_size=BATCH_SIZE)
    keras_preds = np.argmax(keras_probs, axis=1)
    keras_f1 = f1_score(y_test, keras_preds, average='macro')
    keras_acc = accuracy_score(y_test, keras_preds)

    print(f"Keras Macro F1-Score: {keras_f1:.6f}")
    print(f"Keras Accuracy: {keras_acc:.6f}")
    
    # From scratch
    print("\nBuilding and performing prediction with from-scratch model")
    print(f"Batch size: {BATCH_SIZE}")
    
    scratch_model = build_model_from_scratch(keras_model)
    
    all_scratch_preds = []
    num_samples = len(X_test)

    for i in range(0, num_samples, BATCH_SIZE):
        x_batch = X_test[i : i + BATCH_SIZE]

        probs_batch = forward_pass_scratch(x_batch, scratch_model)

        preds_batch = np.argmax(probs_batch, axis=1)
        all_scratch_preds.append(preds_batch)
    
    scratch_preds = np.concatenate(all_scratch_preds)
    
    scratch_f1 = f1_score(y_test, scratch_preds, average='macro')
    scratch_acc = accuracy_score(y_test, scratch_preds)
    
    print(f"From-Scratch Macro F1-Score: {scratch_f1:.6f}")
    print(f"From-Scratch Accuracy: {scratch_acc:.6f}")

    # Comparison
    print("\nComparison Summary")
    np.testing.assert_allclose(keras_f1, scratch_f1, rtol=1e-5, atol=1e-5)
    print("Macro F1-Scores are approximately equal.")
    
    if np.array_equal(keras_preds, scratch_preds):
        print("Predictions are identical.")
    else:
        diff_count = np.sum(keras_preds != scratch_preds)
        total_count = len(y_test)
        print(f"Predictions have {diff_count} differences out of {total_count} samples ({diff_count/total_count:.4%} difference).")

if __name__ == '__main__':
    main()