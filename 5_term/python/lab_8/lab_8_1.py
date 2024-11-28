import sqlite3

conn = sqlite3.connect(
    "C:\\Users\\Бобр Даша\\Desktop\\university\\3 КУРС\\5 сем\\Язык Python для работы с данными\\lab_8\\orders.db"
)
cur = conn.cursor()

cur.execute(
    """CREATE TABLE IF NOT EXISTS sender (
                    sender_code INT PRIMARY KEY,
                    last_name TEXT,
                    first_name TEXT,
                    middle_name TEXT,
                    date_of_birth TEXT,
                    indexx TEXT,
                    town TEXT,
                    street TEXT,
                    house TEXT,
                    apartment TEXT,
                    phone TEXT);"""
)

cur.execute(
    """CREATE TABLE IF NOT EXISTS transport (
                    car_number TEXT PRIMARY KEY,
                    car_brand TEXT,
                    date_of_registration TEXT,
                    color TEXT);"""
)

cur.execute(
    """INSERT INTO sender(sender_code, last_name, first_name, middle_name, date_of_birth, indexx, town, street, house, apartment, phone)
 VALUES ('1', 'Коровкин', 'Сергей', 'Кириллович', '02.02.2002', '236029', 'Калининград', 'Зелёная', '79', '21', '89115553344');"""
)

cur.execute(
    """INSERT INTO sender(sender_code, last_name, first_name, middle_name, date_of_birth, indexx, town, street, house, apartment, phone)
 VALUES ('2', 'Сорокин', 'Михаил', 'Данилович', '01.01.2001', '236028', 'Калининград', 'Озёрная', '58', '2', '89118889977');"""
)

cur.execute("UPDATE sender SET phone = 89112223663 WHERE last_name = 'Коровкин'")

cur.execute(
    """INSERT INTO transport(car_number, car_brand, date_of_registration, color)
 VALUES ('А203АА', 'ауди', '28.12.2022', 'розовый');"""
)

cur.execute(
    """INSERT INTO transport(car_number, car_brand, date_of_registration, color)
 VALUES ('В243ВВ', 'мерседес', '04.11.2019', 'чёрный');"""
)

conn.commit()
conn.close()
