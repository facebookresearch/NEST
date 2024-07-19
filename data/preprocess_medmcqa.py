import json
import jsonlines
import os
import sys
import string
import re

def has_punctuation(word):
    return any(char in string.punctuation for char in word)

mapping = {1:"opa", 2:"opb", 3:"opc", 4:"opd"}

results = []
with open(sys.argv[1]) as f:
    for line in f:
        data = json.loads(line)
        assert data["cop"] > 0
        question = data["question"]
        cop = mapping[data["cop"]]
        answer = data[cop].replace("\n", " ").replace("\t", " ")

        if len(answer.split(" ")) <= 5 and answer != "All of the above" and not has_punctuation(answer) and "figure" not in question.lower():
            results.append({"input":question, "output":[answer]})

with jsonlines.open(sys.argv[2], "w") as f:
    f.write_all(results)