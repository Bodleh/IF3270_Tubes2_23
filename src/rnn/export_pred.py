import tensorflow as tf
import pandas as pd
import dataset
import keras_model
from numpy_rnn import NumpyRNNModel

NUM_LAYERS = 1
UNITS = 32
BIDIRECTIONAL = True
MAX_LEN = 64
WEIGHTS_PATH = "saved/rnn32bi.weights.h5"
VECTORIZER = "saved/vectorizer.keras"
OUT_CSV = "saved/test_predictions.csv"


_ = tf.keras.utils.set_random_seed(42)
train_ds, val_ds, (vect, x_test, y_test) = dataset.prepare_datasets(seed=42)

keras_model = keras_model.build_rnn_model(
    vocab_size=vect.vocabulary_size(),
    num_layers=NUM_LAYERS,
    units=UNITS,
    bidirectional=BIDIRECTIONAL,
    max_len=MAX_LEN,
)
keras_model.load_weights(WEIGHTS_PATH)
y_pred_keras = keras_model.predict(vect(x_test), verbose=0).argmax(1)

np_model = NumpyRNNModel(
    weights_h5=WEIGHTS_PATH,
    vect_path=VECTORIZER,
    num_layers=NUM_LAYERS,
    units=UNITS,
    bidirectional=BIDIRECTIONAL,
    max_len=MAX_LEN
)
y_pred_numpy = np_model.predict(list(x_test))

id2label = {0: "positive", 1: "negative", 2: "neutral"}
df = pd.DataFrame({
    "text": x_test,
    "true_label": [id2label[y] for y in y_test],
    "keras_pred": [id2label[y] for y in y_pred_keras],
    "numpy_pred": [id2label[y] for y in y_pred_numpy],
})

df["correct"] = df["true_label"] == df["keras_pred"]

df.to_csv("saved/test_predictions.csv", index=False)
