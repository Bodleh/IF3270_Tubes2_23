import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, classification_report
import dataset
import keras_model
from numpy_rnn import NumpyRNNModel

NUM_LAYERS = 1
UNITS = 32
BIDIRECTIONAL = True
MAX_LEN = 64
DROPOUT = 0.30
REC_DROPOUT = 0.15
L2_REG = 1e-6

WEIGHTS_PATH = "saved/rnn32bi.weights.h5"
VECTORIZER_DIR = "saved/vectorizer.keras"

_ = tf.keras.utils.set_random_seed(42)
train_ds, val_ds, (vect, x_test, y_test) = dataset.prepare_datasets(seed=42)

# Keras model
keras_model = keras_model.build_rnn_model(
    vocab_size=vect.vocabulary_size(),
    num_layers=NUM_LAYERS,
    units=UNITS,
    bidirectional=BIDIRECTIONAL,
    dropout=DROPOUT,
    rec_dropout=REC_DROPOUT,
    l2_reg=L2_REG,
    max_len=MAX_LEN
)
keras_model.load_weights(WEIGHTS_PATH)

y_pred_keras = keras_model.predict(vect(x_test), verbose=0).argmax(axis=1)

#  NumPy model
np_model = NumpyRNNModel(
    weights_h5=WEIGHTS_PATH,
    vect_path=VECTORIZER_DIR,
    num_layers=NUM_LAYERS,
    units=UNITS,
    bidirectional=BIDIRECTIONAL,
    max_len=MAX_LEN
)

y_pred_numpy = np_model.predict(list(x_test))


print("\nKeras  macro-F1 :", f1_score(y_test, y_pred_keras,  average='macro'))
print("NumPy  macro-F1 :", f1_score(y_test, y_pred_numpy, average='macro'))

print("\nClassification report (NumPy vs ground truth):")
print(classification_report(y_test, y_pred_numpy, digits=4))
