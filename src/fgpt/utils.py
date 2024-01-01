import json 


def write_to_json(data, path):
    json_string = json.dumps(data)
    with open(path, "w") as f:
        f.write(json_string)

def read_from_json(path):
    with open(path, "r") as f:
        json_string = f.read()
    return json.loads(json_string)