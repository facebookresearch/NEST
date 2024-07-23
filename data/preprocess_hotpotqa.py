import sys
import json
import jsonlines

mode = sys.argv[1]
with open(f"datasets/hotpotqa/hotpot_{mode}_fullwiki_v1.json") as f:
    data = json.load(f)

results = []
for d in data:
    answer = d["answer"].lower().strip()
    if "yes" not in answer and "no" not in answer:
        results.append({"input":d["question"], "output":[answer]})


print("saving lines...")
with jsonlines.open(f"datasets/hotpotqa/{mode}.jsonl", "w") as f:
    f.write_all(results)