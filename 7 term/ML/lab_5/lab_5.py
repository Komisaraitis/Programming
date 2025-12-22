import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path


# ЗАГРУЗКА ДАННЫХ

training_file = "C:\\Users\\Бобр Даша\\Desktop\\university\\4 КУРС\\7 семестр\\МО\\lab_5\\training_file.txt"
text = Path(training_file).read_text(encoding="utf-8")

# ПОДГОТОВКА СЛОВАРЯ

vocab = sorted(set(text))
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}
vocab_size = len(vocab)

# ПОДГОТОВКА ДАННЫХ

SEQ_LENGTH = 60
STEP = 3

sentences = []
next_chars = []

for i in range(0, len(text) - SEQ_LENGTH, STEP):
    sentences.append(text[i : i + SEQ_LENGTH])
    next_chars.append(text[i + SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH), dtype=np.int32)
y = np.zeros((len(sentences)), dtype=np.int32)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t] = char_to_idx[char]
    y[i] = char_to_idx[next_chars[i]]

# ПОСТРОЕНИЕ МОДЕЛИ
print("\nПостроение модели LSTM\n ")

EMBEDDING_DIM = 64

model = keras.Sequential(
    [
        layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM),
        layers.LSTM(128),
        layers.Dropout(0.2),
        layers.Dense(vocab_size, activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.002),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

model.summary()

# ОБУЧЕНИЕ МОДЕЛИ
print("\nОбучение модели\n")

EPOCHS = 25
BATCH_SIZE = 256
history = model.fit(
    x, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=1
)


# ФУНКЦИЯ ГЕНЕРАЦИИ
def generate_abracadabra(model, seed_text, num_chars=500, temperature=1.0):
    """Генерирует абракадабру на основе начальной строки."""
    generated = seed_text

    for i in range(num_chars):
        seq = generated[-SEQ_LENGTH:] if len(generated) >= SEQ_LENGTH else generated
        x_pred = np.zeros((1, SEQ_LENGTH), dtype=np.int32)

        start_idx = SEQ_LENGTH - len(seq)
        for t, char in enumerate(seq):
            if char in char_to_idx:
                x_pred[0, start_idx + t] = char_to_idx[char]
            else:
                x_pred[0, start_idx + t] = 0

        preds = model.predict(x_pred, verbose=0)[0]

        preds_tensor = tf.convert_to_tensor(preds, dtype=tf.float32)
        logits = tf.math.log(preds_tensor + 1e-8) / temperature
        next_id = tf.random.categorical(tf.expand_dims(logits, 0), 1)[0, 0]
        next_index = next_id.numpy()

        next_char = idx_to_char[next_index]
        generated += next_char

    return generated


# ГЕНЕРАЦИЯ ТЕКСТОВ
print("\nГенерация абракадабры\n")

seed_text = "Часто прихожу на"
temperature = 0.9

generated = generate_abracadabra(
    model, seed_text=seed_text, num_chars=500, temperature=temperature
)

print(generated[:500])

output_path = "C:\\Users\\Бобр Даша\\Desktop\\university\\4 КУРС\\7 семестр\\МО\\lab_5\\abracadabra.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(generated)

