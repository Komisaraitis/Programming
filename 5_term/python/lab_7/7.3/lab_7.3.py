import json

with open('ex_3.json') as file:
    data = json.load(file)

new_invoice = {
    "id": 3,
    "total": 300.00,
    "items": [
    {
        "name": "item 4",
        "quantity": 1,
        "price": 100.00
    },
    {
        "name": "item 5",
        "quantity": 2,
        "price": 150.00
    }
    ]
}

data['invoices'].append(new_invoice)

with open('new_ex_3.json', 'w') as file:
    json.dump(data, file, indent=4)

