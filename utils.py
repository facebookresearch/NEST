import json

def load_datasets(path, batch_size=16):
    batch_inputs = []
    batch_outputs = []
    with open(path) as f:
        for line in f:
            if len(batch_inputs) == batch_size:
                yield dict(input=batch_inputs, output=batch_outputs)
                batch_inputs, batch_outputs = [], []
            data = json.loads(line)
            batch_inputs.append(data["input"])
            batch_outputs.append(data["output"])
    if len(batch_inputs) > 0:
        yield dict(input=batch_inputs, output=batch_outputs)

def alr(pred, answers):
    has_answer = False
    pred = pred.lower().strip()
    for answer in answers:
        if answer.lower().strip() in pred:
            has_answer = True
        return int(has_answer)
    return int(has_answer)