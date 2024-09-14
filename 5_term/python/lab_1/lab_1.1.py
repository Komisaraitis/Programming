from pathlib import Path
from sys import argv
import shutil

if len(argv) == 1:
    way = "."
else:
    way = argv[1]
flag = True
txt_files = list(Path(way).glob("*.*")) #получение списка файлов
list = []
for c in txt_files:
    if Path(c).stat().st_size <= 2048:
        Path("C:\\small").mkdir(parents=True, exist_ok=True) #создание новой директории
        print(c.name, end="\n")
        shutil.copy(c, "C:\\small\\" + c.name) #копируем файлы в папку small
        flag = False
if flag:
    print("Нет нужных файлов")
