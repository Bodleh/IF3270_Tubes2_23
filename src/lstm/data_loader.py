import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re

def load_data_from_github():
    base_url = "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/indonesian/"
    print(f"Loading data from {base_url}")
    
    try:
        df_train = pd.read_csv(base_url + "train.csv")
        df_val = pd.read_csv(base_url + "valid.csv")
        df_test = pd.read_csv(base_url + "test.csv")
        print("Data loaded successfully.")
        return df_train, df_val, df_test
    except Exception as e:
        print(f"Failed to load data. Error: {e}")
        return None, None, None

def clean_text(text: str):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return text.lower().strip()

def prepare_data(max_vocab_size=10000, max_sequence_length=256, batch_size=32):
    df_train, df_val, df_test = load_data_from_github()
    if df_train is None:
        return None, None, None, None, None, None

    for df in [df_train, df_val, df_test]:
        df['cleaned_text'] = df['text'].apply(clean_text)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(df_train['label'])
    y_val_encoded = label_encoder.transform(df_val['label'])
    y_test_encoded = label_encoder.transform(df_test['label'])
    num_classes = len(label_encoder.classes_)

    X_train, X_val, X_test = df_train['cleaned_text'], df_val['cleaned_text'], df_test['cleaned_text']

    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_vocab_size,
        output_mode='int',
        output_sequence_length=max_sequence_length
    )
    vectorize_layer.adapt(X_train)

    X_train_vec = vectorize_layer(X_train)
    X_val_vec = vectorize_layer(X_val)
    X_test_vec = vectorize_layer(X_test)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train_vec, y_train_encoded)).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val_vec, y_val_encoded)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test_vec, y_test_encoded)).batch(batch_size)

    test_data_for_scratch = (X_test_vec.numpy(), y_test_encoded)

    return train_ds, val_ds, test_ds, vectorize_layer, num_classes, test_data_for_scratch

# # Example
# if __name__ == '__main__':
#     (train_dataset, val_dataset, test_dataset, 
#      text_vectorizer, n_classes, _) = prepare_data()
    
#     if train_dataset:
#         print(f"\nNumber of classes: {n_classes}")
#         print("A batch from the training dataset:")
#         for text_batch, label_batch in train_dataset.take(1):
#             print("Texts (vectorized):", text_batch.numpy())
#             print("Labels (encoded):", label_batch.numpy())
#             print(f"Batch shape: {text_batch.shape}")