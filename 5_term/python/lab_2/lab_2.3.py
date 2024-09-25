from PIL import Image, ImageFilter

with Image.open("C:\\Users\\Бобр Даша\Desktop\\university\\3 КУРС\\5 сем\\Язык Python для работы с данными\\lab_2\\picture.jpg") as img:
    img.load()

with Image.open("C:\\Users\\Бобр Даша\Desktop\\university\\3 КУРС\\5 сем\\Язык Python для работы с данными\\lab_2\\realpython-logo.png") as img_logo:
    img_logo.load()

img_logo = img_logo.convert("L")
threshold = 50
img_logo = img_logo.point(lambda x: 255 if x > threshold else 0)
img_logo = img_logo.resize((img_logo.width // 2, img_logo.height // 2))
img_logo = img_logo.filter(ImageFilter.CONTOUR)
img_logo = img_logo.point(lambda x: 0 if x == 255 else 255)
img.paste(img_logo, (250, 120), img_logo)

img.show()
img.save("C:\\Users\\Бобр Даша\Desktop\\university\\3 КУРС\\5 сем\\Язык Python для работы с данными\\lab_2\\picture_with_logo.jpg")


