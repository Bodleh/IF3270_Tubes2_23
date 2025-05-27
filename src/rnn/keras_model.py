import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers


def build_rnn_model(
    vocab_size: int,
    num_layers: int = 1,
    units: int = 64,
    bidirectional: bool = True,
    dropout: float = 0.3,
    rec_dropout: float = 0.15,
    l2_reg: float = 1e-6,
    max_len: int = 64,
    emb_dim: int = 128,
    lr: float = 1e-3,
):
    inp = layers.Input(shape=(max_len,), dtype="int32")
    x = layers.Embedding(
        vocab_size,
        emb_dim,
        mask_zero=True,
        name="embed"
    )(inp)

    for i in range(num_layers):
        return_seq = (i < num_layers - 1)
        rnn = layers.SimpleRNN(
            units,
            dropout=dropout,
            recurrent_dropout=rec_dropout,
            kernel_regularizer=regularizers.l2(l2_reg),
            return_sequences=return_seq,
            name=f"rnn_{i}"
        )
        x = layers.Bidirectional(rnn, name=f"bi_rnn_{i}")(
            x) if bidirectional else rnn(x)

    x = layers.Dropout(dropout)(x)
    out = layers.Dense(3, activation="softmax", name="classifier")(x)

    model = models.Model(inp, out)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizers.Adam(lr),
        metrics=["accuracy",
                 tf.keras.metrics.SparseCategoricalCrossentropy(name="sce")]
    )
    return model
