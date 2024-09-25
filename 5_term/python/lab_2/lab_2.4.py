from PIL import Image, ImageDraw, ImageFont

# Создание 3 карточек
for i in range(1, 4):
    # Создание нового изображения
    imag = Image.new("RGB", (100, 100), (255, 255, 255))
    
    # Создание объекта для рисования на изображении
    draw = ImageDraw.Draw(imag)
    
    # Добавление рамки
    draw.rectangle((0, 0, 100, 100), outline=(0, 0, 255), width=5)

    fnt = ImageFont.truetype("c:\Windows\Fonts\GOUDYSTO.TTF", 50)
    
    draw.text((30, 20), str(i), font=fnt, fill="red")
    
    # Вывод изображения
    imag.show()
    imag.save("C:\\Users\\Бобр Даша\Desktop\\university\\3 КУРС\\5 сем\\Язык Python для работы с данными\\lab_2\\"+str(i)+".png")
