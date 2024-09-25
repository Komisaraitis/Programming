from PIL import Image
from sys import argv

with Image.open(str(argv[1])) as img:
    img.load()
width, height = img.size

# Создание счетчиков для каждого цвета
red_count = 0
green_count = 0
blue_count = 0

# Подсчет количества пикселей каждого цвета
for x in range(width):
    for y in range(height):
        r, g, b = img.getpixel((x, y))
        red_count += r
        green_count += g
        blue_count += b
# Определение наиболее используемого цвета
max_count = max(red_count, green_count, blue_count)
if max_count == red_count:
    print("Наиболее используемый цвет: Red")
elif max_count == green_count:
    print("Наиболее используемый цвет: Green")
elif max_count == blue_count:
    print("Наиболее используемый цвет: Blue")
else:
    print("Невозможно определить наиболее используемый цвет :(")
