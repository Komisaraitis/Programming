from PIL import Image
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-ftype")
txt_files = list(Path(".").glob("*." + parser.parse_args().ftype))
for c in txt_files:
    with Image.open(c) as img:
        img = img.resize((50, 50))
        img.show()
