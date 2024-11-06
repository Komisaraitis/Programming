import xml.etree.ElementTree as ET

tree = ET.parse(
    "C:\\Users\\Бобр Даша\\Desktop\\university\\3 КУРС\\5 сем\\Язык Python для работы с данными\\lab_6\\ex_3.xml"
)
root = tree.getroot()

items = root.findall(".//СведТов")

for item in items:
    name = item.get("НаимТов")
    quantity = item.get("КолТов")
    price = item.get("ЦенаТов")

    print(f"Наименование товара: {name}, количество: {quantity}, цена: {price}.")

