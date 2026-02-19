import re
import random
import json
from collections import defaultdict
import markovify
import nltk
from nltk.tokenize import word_tokenize
import os

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


class RhymingPoetryGenerator:
    def __init__(self, model_name="poet"):
        self.model_name = model_name
        self.vowels = "аеёиоуыэюя"
        self.markov_model_reversed = None
        self.rhyme_search = None
        self.original_lines = set()

    def _tokenize_russian(self, text):
        """Токенизация"""
        tokens = word_tokenize(text, language="russian")
        return [
            w.strip().lower()
            for w in tokens
            if re.fullmatch(r"[а-яё]+", w, re.IGNORECASE)
        ]

    def _reverse_words(self, line):
        """реверсирование"""
        return " ".join(reversed(self._tokenize_russian(line)))

    def _extract_last_word(self, line):
        """Извлекает последнее слово из строки"""
        words = self._tokenize_russian(line)
        return words[-1] if words else None

    def _count_vowels(self, text):
        """количество гласных букв в тексте"""
        return sum(1 for c in text.lower() if c in self.vowels)

    def load_and_train(self, filepath="poems_clean.txt", state_size=2):
        """обучение моделей"""

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

        for poem in re.split(r"\n\s*\n", content):
            for line in [l.strip() for l in poem.split("\n") if l.strip()]:
                if len(line) < 8 or self._count_vowels(line) < 2:
                    continue
                all_lines.append(line)
                all_words.update(w for w in self._tokenize_russian(line) if len(w) >= 2)

        self.rhyme_search = RhymeSearch()
        self.rhyme_search.train(list(all_words))
        self.rhyme_search.save_json(f"{self.model_name}_rhymes.json")

        reversed_text = "\n".join(self._reverse_words(line) for line in all_lines)

        self.markov_model_reversed = markovify.NewlineText(
            reversed_text,
            state_size=state_size,
            retain_original=False,
            well_formed=False,
        )

        return True

    def _is_unique_line(self, line, threshold=0.85):
        """не является ли сгенерированная строка копией или почти копией строки из датасета"""
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

    def _generate_line_with_rhyme(
        self, target_word, max_attempts=50, min_words=3, max_words=10
    ):
        """Генерирует строку, заканчивающуюся словом, рифмующимся с target_word"""
        if not self.markov_model_reversed or not target_word:
            return None, None
        exclude_words = {target_word}
        for _ in range(max_attempts):
            rhyme_word = self.rhyme_search.give_rhyme(target_word, exclude_words)
            if not rhyme_word:
                return None, None
            exclude_words.add(rhyme_word)
            try:
                rev_line = self.markov_model_reversed.make_sentence_with_start(
                    rhyme_word,
                    strict=False,
                    min_words=min_words,
                    max_words=max_words,
                    test_output=False,
                )
                if not rev_line:
                    continue
                words = self._tokenize_russian(rev_line)
                line = self._postprocess_line(words, min_words)
                if line:
                    return line, rhyme_word
            except Exception:
                continue
        return None, None

    def _generate_free_line(self, max_attempts=40, min_words=4, max_words=10):
        """Генерирует строку без фиксированной рифмы"""
        if not self.markov_model_reversed:
            return None
        for _ in range(max_attempts):
            try:
                rev_line = self.markov_model_reversed.make_sentence(
                    min_words=min_words,
                    max_words=max_words,
                    tries=20,
                    test_output=False,
                )
                if not rev_line:
                    continue
                words = self._tokenize_russian(rev_line)
                line = self._postprocess_line(words, min_words)
                if line:
                    return line
            except Exception:
                continue
        return None

    def _simplify_scheme(self, scheme, length):
        """Приводит схему рифмовки к нужной длине"""
        clean = "".join(c.upper() for c in scheme if c.isalpha()) or "AABB"
        if len(clean) < length:
            clean = (clean * ((length + len(clean) - 1) // len(clean)))[:length]
        return list(clean[:length])

    def generate_poem(self, lines=8, rhyme_scheme="AABB", start_line=None):
        """генерация стихов"""
        if not self.markov_model_reversed or not self.rhyme_search:
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
                line = self._generate_free_line(min_words=4, max_words=9)
                if not line:
                    return None
                poem.append(line)
                if last_word := self._extract_last_word(line):
                    rhyme_map[rhyme_letter] = last_word
            else:
                line, rhyme_word = self._generate_line_with_rhyme(
                    rhyme_map[rhyme_letter], min_words=4, max_words=9
                )
                if not line:
                    return None
                poem.append(line)
                if rhyme_word:
                    rhyme_map[rhyme_letter] = rhyme_word

        return poem[:lines]

    def display_poem(self, poem):
        """вывод стихотворения"""
        print("\nСгенерированное стихотворение\n")
        if not poem:
            print("Не удалось сгенерировать стихотворение")
            return
        for line in poem:
            print(line)


# ЗАПУСК


def main():
    print("ГЕНЕРАЦИЯ СТИХОВ С ПОМОЩЬЮ ЦЕПЕЙ МАРКОВА")
    generator = RhymingPoetryGenerator(model_name="poet")
    if not generator.load_and_train("poems_clean.txt", state_size=2):
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
        print("\nНе удалось сгенерировать идеальное стихотворение за 100 попыток.")
        print("   Попробуйте:")
        print("   • Увеличить количество строк")
        print("   • Выбрать другую схему рифмовки")
        print("   • Изменить начальную строку")


if __name__ == "__main__":
    main()
