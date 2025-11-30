import sys


def main():
    anime_dict = {}
    for line in sys.stdin:
        try:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            anime_id = parts[0]
            score = float(parts[1])
            title = parts[2]

            if anime_id not in anime_dict or score > anime_dict[anime_id][0]:
                anime_dict[anime_id] = (score, title)

        except Exception as e:
            continue

    anime_list = [(score, title) for _, (score, title) in anime_dict.items()]
    anime_list.sort(key=lambda x: x[0], reverse=True)

    total_count = len(anime_list)
    top_10 = anime_list[:10]

    print("Number of adventure anime with rating > 8.0: {}".format(total_count))
    print()
    print("TOP-10 adventure anime with highest ratings:")
    print("Rank Title                                    Rating")
    print("--------------------------------------------------")

    for i, (rating, title) in enumerate(top_10, 1):
        print("{:<4} {:<35} {:<6.2f}".format(i, title, rating))


if __name__ == "__main__":
    main()
