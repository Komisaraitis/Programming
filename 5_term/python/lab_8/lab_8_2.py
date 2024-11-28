from peewee import *

db = SqliteDatabase(
    "C:\\Users\\Бобр Даша\\Desktop\\university\\3 КУРС\\5 сем\\Язык Python для работы с данными\\lab_8\\orders.db"
)


class sender(Model):
    sender_code = IntegerField(primary_key=True)
    last_name = TextField()
    first_name = TextField()
    middle_name = TextField()
    date_of_birth = TextField()
    indexx = TextField()
    town = TextField()
    street = TextField()
    house = TextField()
    apartment = TextField()
    phone = TextField()

    class Meta:
        database = db


class transport(Model):
    car_number = TextField(primary_key=True)
    car_brand = TextField()
    date_of_registration = TextField()
    color = TextField()

    class Meta:
        database = db


class receiver(Model):
    receiver_code = IntegerField(primary_key=True)
    last_name = TextField()
    first_name = TextField()
    middle_name = TextField()
    date_of_birth = TextField()
    indexx = TextField()
    town = TextField()
    street = TextField()
    house = TextField()
    apartment = TextField()
    phone = TextField()

    class Meta:
        database = db


class сourier(Model):
    courier_code = IntegerField(primary_key=True)
    last_name = TextField()
    first_name = TextField()
    middle_name = TextField()
    passport_number = TextField()
    date_of_birth = TextField()
    date_of_employment = TextField()
    start_of_the_working_day = TextField()
    end_of_the_working_day = TextField()
    town = TextField()
    street = TextField()
    house = TextField()
    apartment = TextField()
    phone = TextField()

    class Meta:
        database = db


class order(Model):
    order_code = IntegerField(primary_key=True)
    sender = ForeignKeyField(sender)
    receiver = ForeignKeyField(receiver)
    order_date = TextField()
    delivery_date = TextField()
    delivery_price = TextField()
    courier = ForeignKeyField(сourier)
    transport = ForeignKeyField(transport)

    class Meta:
        database = db


receiver.create_table()
сourier.create_table()
order.create_table()

receiver.create(
    receiver_code="1",
    last_name="Мушкина",
    first_name="Елена",
    middle_name="Александровна",
    date_of_birth="03.03.2003",
    indexx="256427",
    town="Москва",
    street="Морская",
    house="59",
    apartment="157",
    phone="84758954721",
)
receiver.create(
    receiver_code="2",
    last_name="Малов",
    first_name="Игорь",
    middle_name="Андреевич",
    date_of_birth="04.07.1995",
    indexx="244325",
    town="Казань",
    street="Горького",
    house="65",
    apartment="6",
    phone="88438114431",
)
сourier.create(
    courier_code="1",
    last_name="Сидоров",
    first_name="Артём",
    middle_name="Викторович",
    passport_number="729721",
    date_of_birth="21.05.1983",
    date_of_employment="02.03.2023",
    start_of_the_working_day="8:00",
    end_of_the_working_day="20:00",
    town="Пионерский",
    street="Вокзальная",
    house="9",
    apartment="31",
    phone="89112453698",
)
сourier.create(
    courier_code="2",
    last_name="Петров",
    first_name="Алексей",
    middle_name="Викторович",
    passport_number="539481",
    date_of_birth="13.04.1999",
    date_of_employment="20.07.2024",
    start_of_the_working_day="9:00",
    end_of_the_working_day="21:00",
    town="Зеленоградск",
    street="Гагарина",
    house="78",
    apartment="22",
    phone="89095887412",
)
order.create(
    order_code="1",
    sender="1",
    receiver="1",
    order_date="22.09.2024",
    delivery_date="29.09.2024",
    delivery_price="5900",
    courier="1",
    transport="А203АА",
)
order.create(
    order_code="2",
    sender="2",
    receiver="2",
    order_date="28.10.2024",
    delivery_date="05.11.2024",
    delivery_price="1249",
    courier="2",
    transport="В243ВВ",
)
