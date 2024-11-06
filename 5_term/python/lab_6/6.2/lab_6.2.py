import xml.etree.ElementTree as ET

tree = ET.parse(
    "C:\\Users\\Бобр Даша\\Desktop\\university\\3 КУРС\\5 сем\\Язык Python для работы с данными\\lab_6\\ex_2.xml"
)
root = tree.getroot()

item = ET.SubElement(root.find("Detail"), "Item")

artname = ET.SubElement(item, "ArtName")
artname.text = "Сыр Пармезан"
barcode = ET.SubElement(item, "Barcode")
barcode.text = "2000000000063"
QNT = ET.SubElement(item, "QNT")
QNT.text = "234,5"
QNTPack = ET.SubElement(item, "QNTPack")
QNTPack.text = "234,5"
unit = ET.SubElement(item, "Unit")
unit.text = "шт"
SN1 = ET.SubElement(item, "SN1")
SN1.text = "00000020"
SN2 = ET.SubElement(item, "SN2")
SN2.text = "09.10.2019"
QNTRows = ET.SubElement(item, "QNTRows")
QNTRows.text = "17"

items = root.findall(".//Detail/Item")

count_qnt = 0
count_qntRows = 0

for item in items:
    qnt = float(item.find("QNT").text.replace(",", "."))
    count_qnt += qnt
    qntRows = int(item.find("QNTRows").text)
    count_qntRows += qntRows

summary = root.find(".//Summary")

summary.find("Summ").text = str(count_qnt).replace(".", ",")
summary.find("SummRows").text = str(count_qntRows)

tree = ET.ElementTree(root)
tree.write(
    "C:\\Users\\Бобр Даша\\Desktop\\university\\3 КУРС\\5 сем\\Язык Python для работы с данными\\lab_6\\new_ex_2.xml",
    encoding="utf-8",
    xml_declaration=True,
)
