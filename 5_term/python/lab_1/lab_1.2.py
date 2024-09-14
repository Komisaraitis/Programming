from pathlib import Path
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dirpath", default=".")
parser.add_argument("--files", nargs="*")

way = parser.parse_args().dirpath
files = parser.parse_args().files

if not files:
    txt_files = list(Path(way).glob("*.*"))
    print("Количество файлов:", len(txt_files))
    count = 0
    for c in txt_files:
        count += Path(c).stat().st_size
    print("Общий размер файлов:", count)
else:
    yes_files = []
    no_files = []
    for c in files:
        if os.path.exists(
            os.path.join(way, c)
        ):  # проверяет наличие переданных файлов в папке
            yes_files.append(c)
        else:
            no_files.append(c)
    with open("C:\\yes_files.txt", "w") as yes_file, open(
        "C:\\no_files.txt", "w"
    ) as no_file:
        yes_file.write("\n".join(yes_files))
        no_file.write("\n".join(no_files))
    print("Файлы, которые присутствуют в папке:\n", "\n".join(yes_files))
    print("Файлы, которые отсутствуют в папке:\n", "\n".join(no_files))
