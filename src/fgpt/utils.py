import json 


def write_to_json(data, path):
    json_string = json.dumps(data)
    with open(path, "w") as f:
        f.write(json_string)
