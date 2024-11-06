import xmlschema
import xml.etree.ElementTree as ET

schema = xmlschema.XMLSchema(
    "C:\\Users\\Бобр Даша\\Desktop\\university\\3 КУРС\\5 сем\\Язык Python для работы с данными\\lab_6\\6.1\\scheme.xsd"
)
tree = ET.parse(
    "C:\\Users\\Бобр Даша\\Desktop\\university\\3 КУРС\\5 сем\\Язык Python для работы с данными\\lab_6\\6.1\\ex_1_error.xml"
)

if schema.is_valid(tree):
    print("Файл прошёл валидацию")
else:
    print("Файл не прошёл валидацию")
