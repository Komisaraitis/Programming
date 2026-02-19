import re, json, random, numpy as np, tensorflow as tf, os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from nltk.tokenize import word_tokenize
from collections import defaultdict

import nltk

nltk.download("punkt_tab", quiet=True)

# КЛАСС ДЛЯ ПОИСКА РИФМ


class RhymeSearch:
    """Поиск рифм по окончаниям"""

    def __init__(self):
        self.rhymes_dict = defaultdict(list)

    def train(self, words):
        """обучение словаря"""
        for word in words:
            clean = self._clean_word(word)
            if len(clean) < 2:
                continue
            if len(clean) >= 3:
                suffix3 = clean[-3:]
                if word not in self.rhymes_dict[suffix3]:
                    self.rhymes_dict[suffix3].append(word)
            suffix2 = clean[-2:]
            if word not in self.rhymes_dict[suffix2]:
                self.rhymes_dict[suffix2].append(word)

    def _clean_word(self, word):
        """очистка слова"""
        return re.sub(r"[^\w\-]", "", word.lower()).strip()

    def give_rhyme(self, word, exclude_words=None):
        """поиск рифмы"""
        if exclude_words is None:
            exclude_words = set()
        clean = self._clean_word(word)
        if len(clean) < 2:
            return None
        candidates = []
        if len(clean) >= 3:
            suffix3 = clean[-3:]
            candidates = [
                w
                for w in self.rhymes_dict.get(suffix3, [])
                if w not in exclude_words and self._clean_word(w) != clean
            ]
        if len(candidates) < 3:
            suffix2 = clean[-2:]
            candidates.extend(
                [
                    w
                    for w in self.rhymes_dict.get(suffix2, [])
                    if w not in exclude_words and self._clean_word(w) != clean
                ]
            )
        candidates = [w for w in set(candidates) if self._clean_word(w) != clean]
        return random.choice(candidates) if candidates else None

    def save_json(self, path):
        """сохранение словаря"""
        data = {k: v for k, v in self.rhymes_dict.items() if len(v) > 1}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ГЕНЕРАТОР СТИХОВ


class LSTMRhymingPoetryGenerator:
    """Генератор стихов с контролем рифмы"""

    def __init__(self, model_name="poet"):
        self.model_name = model_name
        self.vowels = "аеёиоуыэюя"
        self.model = None
        self.rhyme_search = None
        self.vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.id2word = {0: "<pad>", 1: "<unk>", 2: "<bos>", 3: "<eos>"}
        self.VOCAB_SIZE = 4
        self.original_lines = set()
        self.poems = []

    def _tokenize_russian(self, text):
        """Токенизация"""
        tokens = word_tokenize(text, language="russian")
        return [
            w.strip().lower()
            for w in tokens
            if re.fullmatch(r"[а-яё]+", w, re.IGNORECASE)
        ]

    def _reverse_words(self, line):
        """Реверсирование строки"""
        return " ".join(reversed(self._tokenize_russian(line)))

    def _extract_last_word(self, line):
        """Извлекает последнее слово из строки"""
        words = self._tokenize_russian(line)
        return words[-1] if words else None

    def _count_vowels(self, text):
        """Подсчёт гласных"""
        return sum(1 for c in text.lower() if c in self.vowels)

    def _is_unique_line(self, line, threshold=0.85):
        """Проверка уникальности"""
        if not line:
            return False
        words_list = self._tokenize_russian(line)
        words = set(words_list)
        if len(words) < 3:
            return True
        for orig in self.original_lines:
            orig_words_list = self._tokenize_russian(orig)
            orig_words = set(orig_words_list)
            if words_list == orig_words_list:
                return False
            if words and len(words & orig_words) / len(words) >= threshold:
                return False
        return True

    def _postprocess_line(self, words, min_words):
        """Постобработка сгенерированной строки"""
        if len(words) < min_words:
            return None
        normal_line = " ".join(reversed(words))
        if len(normal_line) < 10 or self._count_vowels(normal_line) < 2:
            return None
        if not self._is_unique_line(normal_line, threshold=0.85):
            return None
        if normal_line and normal_line[0].isalpha():
            normal_line = normal_line[0].upper() + normal_line[1:]
        return normal_line

    def load_and_train(self, filepath="poems_clean.txt", epochs=60):
        """обучение моделей"""

        model_path = f"{self.model_name}_model.keras"
        metadata_path = f"{self.model_name}_metadata.json"
        rhymes_path = f"{self.model_name}_rhymes.json"

        if (
            os.path.exists(model_path)
            and os.path.exists(metadata_path)
            and os.path.exists(rhymes_path)
        ):
            print(f"Найдена сохранённая модель, загружаем...")

            self.model = load_model(model_path)

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                self.vocab = metadata["vocab"]
                self.id2word = {int(k): v for k, v in metadata["id2word"].items()}
                self.VOCAB_SIZE = metadata["vocab_size"]

            self.rhyme_search = RhymeSearch()
            with open(rhymes_path, "r", encoding="utf-8") as f:
                rhymes_data = json.load(f)
                self.rhyme_search.rhymes_dict = defaultdict(list, rhymes_data)

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            self.original_lines = {
                line.strip()
                for line in content.split("\n")
                if line.strip() and len(line.strip()) > 10
            }

            poems = [p.strip() for p in content.split("\n\n") if p.strip()]
            self.poems = poems

            print(f"Модель загружена. Размер словаря: {self.VOCAB_SIZE}")
            return True

        if not os.path.exists(filepath):
            print(f"Файл '{filepath}' не найден!")
            return False

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        self.original_lines = {
            line.strip()
            for line in content.split("\n")
            if line.strip() and len(line.strip()) > 10
        }

        all_lines, all_words = [], set()
        poems = [p.strip() for p in content.split("\n\n") if p.strip()]
        self.poems = poems

        for poem in poems:
            lines = [l.strip() for l in poem.split("\n") if l.strip()]
            for line in lines:
                tokens = self._tokenize_russian(line)
                vowel_count = self._count_vowels(line)
                if 3 <= len(tokens) <= 15 and vowel_count >= 2:
                    all_lines.append(line)
                    all_words.update(w for w in tokens if len(w) >= 2)

        for word in all_words:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        self.id2word = {v: k for k, v in self.vocab.items()}
        self.VOCAB_SIZE = len(self.vocab)

        self.rhyme_search = RhymeSearch()
        self.rhyme_search.train(list(all_words))
        self.rhyme_search.save_json(f"{self.model_name}_rhymes.json")

        reversed_encoded = []
        for line in all_lines:
            tokens = self._tokenize_russian(line)
            ids = (
                [self.vocab["<bos>"]]
                + [self.vocab.get(w, self.vocab["<unk>"]) for w in reversed(tokens)]
                + [self.vocab["<eos>"]]
            )
            reversed_encoded.append(ids)

        random.shuffle(reversed_encoded)
        batches = []
        for i in range(0, len(reversed_encoded), 128):
            batch = reversed_encoded[i : i + 128]
            maxlen = max(len(x) for x in batch)
            x, y = [], []
            for seq in batch:
                pad = [0] * (maxlen - len(seq))
                x.append(seq[:-1] + pad)
                y.append(seq[1:] + pad)
            batches.append((np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)))

        print(f"\nОбучение модели ({epochs} эпох)...")
        self.model = self._build_model()

        for epoch in range(epochs):
            total_loss = 0
            total_batches = 0
            for x_batch, y_batch in batches:
                result = self.model.train_on_batch(x_batch, y_batch)
                loss_value = result[0] if isinstance(result, (list, tuple)) else result
                total_loss += loss_value
                total_batches += 1
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / total_batches if total_batches > 0 else 0
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self.model.save(f"{self.model_name}_model.keras")
        metadata = {
            "vocab": self.vocab,
            "id2word": self.id2word,
            "vocab_size": self.VOCAB_SIZE,
        }
        with open(f"{self.model_name}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return True

    def _build_model(
        self, embedding_dim=256, hidden_size=512, num_layers=3, dropout=0.3
    ):
        """Создание архитектуры модели"""
        inputs = Input(shape=(None,))
        x = Embedding(self.VOCAB_SIZE, embedding_dim, mask_zero=True)(inputs)

        for i in range(num_layers):
            x = LSTM(
                hidden_size,
                return_sequences=True,
                dropout=dropout if i < num_layers - 1 else 0.0,
            )(x)

        outputs = Dense(self.VOCAB_SIZE, activation="linear")(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=["sparse_categorical_accuracy"],
        )
        return model

    def _generate_line_with_rhyme(
        self, target_word, max_attempts=50, min_words=4, max_words=9, temperature=0.85
    ):
        """Генерация строки с рифмой"""
        if not self.model or not target_word or target_word not in self.vocab:
            return None, None

        exclude_words = {target_word}
        for _ in range(max_attempts):
            rhyme_word = self.rhyme_search.give_rhyme(target_word, exclude_words)
            if not rhyme_word or rhyme_word not in self.vocab:
                return None, None

            exclude_words.add(rhyme_word)
            tokens, words = [self.vocab["<bos>"], self.vocab[rhyme_word]], [rhyme_word]

            while len(words) < max_words:
                logits = self.model.predict(
                    np.array([tokens], dtype=np.int32), verbose=0
                )[0, -1, :]
                probs = tf.nn.softmax(logits / temperature).numpy()
                next_token = np.random.choice(len(probs), p=probs)
                next_word = self.id2word.get(next_token, "<unk>")

                if next_word in ["<eos>", "<pad>", "<bos>", "<unk>"]:
                    if len(words) >= min_words:
                        break
                    continue

                tokens.append(next_token)
                words.append(next_word)
                if len(words) >= min_words:
                    break

            if len(words) < min_words:
                continue

            line = self._postprocess_line(words, min_words)
            if line:
                return line, rhyme_word

        return None, None

    def _generate_free_line(
        self, max_attempts=40, min_words=4, max_words=9, temperature=0.85
    ):
        """Генерация свободной строки"""
        if not self.model:
            return None

        for _ in range(max_attempts):
            valid_words = [
                w for w in self.vocab if w not in ["<pad>", "<unk>", "<bos>", "<eos>"]
            ]
            if not valid_words:
                return None

            start_word = random.choice(valid_words)
            line, _ = self._generate_line_with_rhyme(
                start_word,
                min_words=min_words,
                max_words=max_words,
                temperature=temperature,
            )
            if line:
                return line

        return None

    def _simplify_scheme(self, scheme, length):
        """Приведение схемы рифмовки к нужной длине"""
        clean = "".join(c.upper() for c in scheme if c.isalpha()) or "AABB"
        if len(clean) < length:
            clean = (clean * ((length + len(clean) - 1) // len(clean)))[:length]
        return list(clean[:length])

    def generate_poem(
        self, lines=8, rhyme_scheme="AABB", start_line=None, temperature=0.85
    ):
        """Генерация стихотворения"""
        if not self.model or not self.rhyme_search:
            print("Модели не обучены. Вызовите load_and_train() сначала.")
            return []

        scheme = self._simplify_scheme(rhyme_scheme, lines)
        poem, rhyme_map = [], {}

        start_idx = 0
        if start_line and len(start_line.strip()) >= 10:
            clean_start = " ".join(self._tokenize_russian(start_line.strip()))
            if clean_start and (last_word := self._extract_last_word(clean_start)):
                poem.append(clean_start)
                rhyme_map[scheme[0]] = last_word
                start_idx = 1

        for i in range(start_idx, lines):
            rhyme_letter = scheme[i]
            if rhyme_letter not in rhyme_map:
                line = self._generate_free_line(
                    min_words=4, max_words=9, temperature=temperature
                )
                if not line:
                    return None
                poem.append(line)
                if last_word := self._extract_last_word(line):
                    rhyme_map[rhyme_letter] = last_word
            else:
                line, rhyme_word = self._generate_line_with_rhyme(
                    rhyme_map[rhyme_letter],
                    min_words=4,
                    max_words=9,
                    temperature=temperature,
                )
                if not line:
                    return None
                poem.append(line)
                if rhyme_word:
                    rhyme_map[rhyme_letter] = rhyme_word

        return poem[:lines]

    def display_poem(self, poem):
        """Вывод стихотворения"""
        print("\nСгенерированное стихотворение\n")
        if not poem:
            print("Не удалось сгенерировать стихотворение")
            return
        for line in poem:
            print(line)


# ЗАПУСК


def main():
    print("ГЕНЕРАЦИЯ СТИХОВ С ПОМОЩЬЮ LSTM")
    generator = LSTMRhymingPoetryGenerator(model_name="poet")

    if not generator.load_and_train("poems_clean.txt", epochs=60):
        print("\nОшибка обучения. Проверьте наличие файла poems_clean.txt")
        return

    start_line = input(
        "\nВведите начальную строку (Enter для случайной генерации):\n> "
    ).strip()

    schemes = {
        "1": ("AABB", "Парная рифма"),
        "2": ("ABAB", "Перекрестная рифма"),
        "3": ("ABBA", "Опоясывающая рифма"),
        "4": ("AAAA", "Монорифма"),
    }

    print("\nВыберите схему рифмовки:")
    for key, (scheme, desc) in schemes.items():
        print(f"   {key}. {desc} ({scheme})")

    choice = input("Ваш выбор [1-4, по умолчанию 1]: ").strip() or "1"
    rhyme_scheme, _ = schemes.get(choice, schemes["1"])

    lines_input = input("Количество строк [4-16, по умолчанию 8]: ").strip() or "8"
    lines = max(4, min(16, int(lines_input))) if lines_input.isdigit() else 8

    poem = None
    for attempt in range(100):
        if attempt % 10 == 0 and attempt > 0:
            print(f"   Попытка {attempt}...")
        poem = generator.generate_poem(
            lines=lines, rhyme_scheme=rhyme_scheme, start_line=start_line or None
        )
        if poem and len(poem) >= 2:
            break

    if poem and len(poem) >= 2:
        generator.display_poem(poem)
    else:
        print("\nНе удалось сгенерировать стихотворение за 100 попыток.")
        print("   Попробуйте:")
        print("   • Увеличить количество строк")
        print("   • Выбрать другую схему рифмовки")
        print("   • Изменить начальную строку")


if __name__ == "__main__":
    main()
