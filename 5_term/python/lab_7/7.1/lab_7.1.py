import json
from jsonschema import validate, exceptions

with open("ex_1_error.json") as file, open("scheme.json") as schem:
    data = json.load(file)
    schema = json.load(schem)

try:
    validate(instance=data, schema=schema)
    print("JSON файл валиден.")
except exceptions.ValidationError as err:
    print(f"Ошибка валидации: {err.message}")
