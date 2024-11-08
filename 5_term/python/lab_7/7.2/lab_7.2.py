import json

with open("new_ex_2.json") as file:
    users = json.load(file)

user_inf = {user["name"]: user["phoneNumber"] for user in users["users"]}

print(user_inf)
