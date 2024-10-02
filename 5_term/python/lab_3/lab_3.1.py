import argparse
import numpy as np
from pathlib import Path
from skimage import transform, color, io

parser = argparse.ArgumentParser()
parser.add_argument("--dirpath", required=True)

ways = list(Path(parser.parse_args().dirpath).glob("*.*"))

print(
    "Введите номер(а) преобразования(ий), которое(ые) Вы хотите сделать. Если преобразований больше одного, вводите номера через пробел:"
)
print("1: поворот на 180 градусов")
print("2: поворот на 90 градусов")
print("3: поворот на 45 градусов")
print("4: преобразование в черно-белое изображение")
print("5: уменьшение размера изображения в два раза")
print("6: комплексное (2 и 5)")
transformation = input().split()

count = 19
for a in transformation:
    for b in ways:
        photo = io.imread(b)
        count += 1
        if a == "1":
            new_photo = transform.rotate(photo, angle=180)
        elif a == "2":
            new_photo = transform.rotate(photo, angle=90)
        elif a == "3":
            new_photo = transform.rotate(photo, angle=45)
        elif a == "4":
            rgb_photo = photo[:, :, :3]
            new_photo = color.rgb2gray(rgb_photo)
        elif a == "5":
            new_photo = transform.resize(
                photo, (photo.shape[0] // 2, photo.shape[1] // 2)
            )
        else:
            new_photo = transform.rotate(photo, angle=90)
            new_photo = transform.resize(
                new_photo,
                (new_photo.shape[0] // 2, new_photo.shape[1] // 2),
            )
        new_photo = (new_photo * 255).astype(np.uint8)
        io.imsave(parser.parse_args().dirpath + "\\" + str(count) + ".png", new_photo)
