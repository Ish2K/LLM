import json

with open('./data/data.jsonl', 'r') as json_file:
    json_list = list(json_file)

messages = []

for json_str in json_list:
    result = json.loads(json_str)
    prompt = result['prompt']
    completion = result['completion']

    d0 = {"role":"system", "content":"Hello, cutie!"}
    d1 = {"role":"user", "content":prompt}
    d2 = {"role":"assistant", "content":completion}

    m = [d0, d1, d2]

    d4 = {"messages":m}
    messages.append(d4)

# dump to jsonl file

with open('./data/data1.jsonl', 'w') as json_file:
    for message in messages:
        json.dump(message, json_file)
        json_file.write('\n')
    