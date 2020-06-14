import json


def load_from_file(filename: str) -> dict:
    with open(filename, 'r', encoding='UTF-8') as file:
        data = file.read()
        return json.loads(data)


def save_to_file(filename: str, data: dict) -> None:
    with open(filename, 'w+', encoding='UTF-8') as file:
        file.write(json.dumps(data, ensure_ascii=False))

