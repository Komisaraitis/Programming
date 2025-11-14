import csv
import io


def reader(file_path: str, chunk_size=1024 * 1024):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        remaining = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                if remaining:
                    yield remaining
                break
            chunk = remaining + chunk
            lines = chunk.split("\n")
            remaining = lines.pop()
            for line in lines:
                yield line
        if remaining:
            yield remaining


def mapper(line: str):
    try:
        if not line.strip() or "anime_id" in line:
            return

        reader = csv.reader(io.StringIO(line))
        row = next(reader)

        if len(row) < 13 or not row[1].isdigit():
            return

        anime_id = row[1]
        genres = row[12].lower() if len(row) > 12 else ""
        title = row[5] if len(row) > 5 else "Unknown"
        score_str = row[8] if len(row) > 8 else "0"

        if "adventure" not in genres:
            return

        try:
            score = float(score_str) if score_str.strip() else 0.0
        except ValueError:
            return

        if score > 8.0:
            yield anime_id, (title, score)

    except Exception:
        return


def shuffler(lst):
    prev_key = None
    buffer = []
    is_new = True

    for key, value in lst:
        if is_new:
            buffer = [value]
            is_new = False
        elif key == prev_key or prev_key is None:
            buffer.append(value)
        else:
            yield prev_key, buffer
            buffer = [value]

        prev_key = key

    if buffer:
        yield prev_key, buffer


def reducer(lst):
    for anime_id, values in lst:
        if values:
            title, score = values[0]
            yield -score, (title, len(values))


def main():
    file_path = "C:\\Users\\Бобр Даша\\Desktop\\university\\4 КУРС\\7 семестр\\BD\\лабы\\lab_1\\final_animedataset.csv"

    all_mapped_data = []
    processed_count = 0

    for line in reader(file_path):
        processed_count += 1
        mapped_data = list(mapper(line))
        all_mapped_data.extend(mapped_data)



    sorted_data = sorted(all_mapped_data, key=lambda x: x[0])

    shuffled_data = list(shuffler(sorted_data))

    reduced_data = list(reducer(shuffled_data))

    final_results = sorted(reduced_data)

    total_anime = len(final_results)
    print(f"Количество аниме в жанре приключения с оценкой > 8.0: {total_anime}")

    if final_results:
        print(f"\nТОП-10 аниме в жанре приключения с наивысшими оценками:")
        print(f"{'Ранг':<4} {'Название':<35} {'Оценка':<8}")
        print("-" * 50)

        for rank, (neg_score, (title, records_count)) in enumerate(
            final_results[:10], 1
        ):
            score = -neg_score
            short_title = title[:32] + "..." if len(title) > 35 else title
            print(f"{rank:<4} {short_title:<35} {score:<8.2f}")


if __name__ == "__main__":
    main()
