import sys


def parse_csv_line(line):
    parts = []
    current = ""
    in_quotes = False

    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == "," and not in_quotes:
            parts.append(current.strip())
            current = ""
        else:
            current += char

    parts.append(current.strip())
    return parts


def main():
    first_line = True

    for line in sys.stdin:
        try:
            line = line.strip()
            if not line:
                continue

            if first_line:
                first_line = False
                continue

            parts = parse_csv_line(line)

            if len(parts) < 13:
                continue

            anime_id = parts[1]
            title = parts[5]
            score_str = parts[8]
            genres = parts[12]

            genres_clean = genres.replace('"', "").replace("'", "").strip().lower()

            genre_list = [genre.strip() for genre in genres_clean.split(",")]

            has_adventure = any(genre == "adventure" for genre in genre_list)

            if has_adventure:
                try:
                    score = float(score_str)
                    if score > 8.0:
                        clean_title = title.strip().replace('"', "")

                        print(f"{anime_id}\t{score:.3f}\t{clean_title}")
                except ValueError:
                    continue

        except Exception as e:
            continue


if __name__ == "__main__":
    main()
