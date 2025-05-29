import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from data_loader import prepare_data
from lstm_keras import build_keras_lstm_model, train_model
from lstm_scratch import NumpyLSTM

def main():    
    BATCH_SIZE = 32
    
    train_ds, val_ds, _, _, NUM_CLASSES, (X_test, y_test) = prepare_data(
        max_vocab_size=15000, 
        max_sequence_length=250,
        batch_size=BATCH_SIZE
    )
    
    # Keras
    MODEL_NAME = 'test_lstm_forward_pass'
    MODEL_CONFIG = [('bidirectional', 64), ('unidirectional', 128)]
    
    print(f"Loading Keras model '{MODEL_NAME}'...")
    keras_model = build_keras_lstm_model(
        num_classes=NUM_CLASSES,
        vocab_size=15000,
        emb_dim=128,
        seq_len=250,
        lstm_layers_config=MODEL_CONFIG
    )
    
    train_model(keras_model, train_ds, val_ds, 10, MODEL_NAME)
    
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
    
    scratch_model = NumpyLSTM(keras_model)
    scratch_preds = scratch_model.predict(X_test, batch_size=BATCH_SIZE)
    
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