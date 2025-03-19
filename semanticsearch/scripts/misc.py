import json


def read_jsonl(filepath):
    """
    Reads a JSON Lines (jsonl) file and returns a list of dictionaries.

    Args:
        filepath (str): The path to the JSONL file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a line in the file.
    """
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    json_object = json.loads(line)
                    data.append(json_object)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line.strip()}. Error: {e}")

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")

    return data
