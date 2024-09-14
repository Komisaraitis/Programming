from pathlib import Path
from sys import argv
import os

if len(argv) == 1:
    way = "."
else:
    way = argv[1]

with open("C:\\no_files.txt", "r") as no_file:
    files = no_file.readlines()
for c in files:
    with open(os.path.join(way, c.strip()), "w") as file:
        file.write('')