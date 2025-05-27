import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import TextVectorization
import random

MAX_VOCAB = 20_000
MAX_LEN = 64
NUM_CLASSES = 3
BATCH_SIZE = 64


def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_raw():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "valid.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    LABEL_MAP = {"positive": 0, "negative": 1, "neutral": 2}
    x_train = train_df["text"].astype(str).to_numpy()
    y_train = train_df["label"].map(LABEL_MAP).to_numpy(dtype=int)

    x_val = val_df["text"].astype(str).to_numpy()
    y_val = val_df["label"].map(LABEL_MAP).to_numpy(dtype=int)

    x_test = test_df["text"].astype(str).to_numpy()
    y_test = test_df["label"].map(LABEL_MAP).to_numpy(dtype=int)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def prepare_datasets(seed: int = 42):
    set_global_seed(seed)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_raw()

    vect = TextVectorization(
        max_tokens=MAX_VOCAB,
        output_mode="int",
        output_sequence_length=MAX_LEN
    )
    vect.adapt(x_train)

    def make_ds(x, y, shuffle=False):
        x_tok = vect(x)
        y_int = tf.convert_to_tensor(y, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((x_tok, y_int))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(x), seed=seed,
                            reshuffle_each_iteration=True)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(x_train, y_train, shuffle=True)
    val_ds = make_ds(x_val,   y_val,   shuffle=False)
    return train_ds, val_ds, (vect, x_test, y_test)
