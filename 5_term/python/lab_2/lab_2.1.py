from PIL import Image

with Image.open("C:\\Users\\Бобр Даша\Desktop\\university\\3 КУРС\\5 сем\\Язык Python для работы с данными\\lab_2\\picture.jpg") as img:
    img.load()
img.convert("RGB")
red, green, blue = img.split()
zeroed_band = red.point(lambda _: 0)

red_merge = Image.merge("RGB", (red, zeroed_band, zeroed_band))
green_merge = Image.merge("RGB", (zeroed_band, green, zeroed_band))
blue_merge = Image.merge("RGB", (zeroed_band, zeroed_band, blue))

img.show()
red_merge.show()
green_merge.show()
blue_merge.show()

