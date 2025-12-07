import subprocess
import sys
import os

PYTHON = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(script_name: str) -> None:
    path = os.path.join(BASE_DIR, script_name)
    if not os.path.exists(path):
        print(f"Файл не найден: {script_name}")
        return

    print(f"\n>>> Запуск {script_name}...\n")
    subprocess.call([PYTHON, path])
    print(f"\n<<< Завершён {script_name}\n")


def main_menu() -> None:
    db_path = os.path.join(BASE_DIR, "search.db")

    print("Мини-поисковик")

    while True:
        print("\n1. Парсинг и создание БД")
        print("2. PageRank (MapReduce)")
        print("3. PageRank (Pregel)")
        print("4. Поиск по документам")
        print("0. Выход")

        choice = input("\nВыбор: ").strip()

        if choice == "1":
            run_script("parser.py")

        elif choice == "2":
            if os.path.exists(db_path):
                run_script("pagerank_mapreduce.py")
            else:
                print("Сначала создайте БД (пункт 1)")

        elif choice == "3":
            if os.path.exists(db_path):
                run_script("pagerank_pregel.py")
            else:
                print("Сначала создайте БД (пункт 1)")

        elif choice == "4":
            if os.path.exists(db_path):
                run_script("search.py")
            else:
                print("Сначала создайте БД (пункт 1)")

        elif choice == "0":
            print("Выход")
            break


if __name__ == "__main__":
    main_menu()
