import sys
import json
import jsonlines
import collections
from tqdm import tqdm
from transformers import AutoTokenizer

def is_title(line):
    if line.count("=") == 2:
        return True
    else:
        return False

mode = sys.argv[1]
mode_map = {"valid":"dev", "test":"test"}

context_len=128
completion_len=256
results = collections.defaultdict(list)
tokenizer = AutoTokenizer.from_pretrained("Trelis/Llama-2-7b-chat-hf-hosted-inference-8bit")

print("loading lines...")
with open(f"datasets/wikitext-103/wiki.{mode}.tokens", "r", encoding="utf-8") as f:
    print("processing lines...")
    lines = f.readlines()

results = {}
title = None
text = ""
for line in lines:
    if len(line.strip()) == 0:
        continue    
    if is_title(line):
        if title is not None:
            results[title] = text
        title = line.replace("=", "").strip()
        text = ""
    else:
        text += line.strip()
        results[title] = text

to_save = []
context_avg_len = 0
target_avg_len = 0
for title, text in tqdm(results.items()):
    text = " ".join([title, text])
    tokens = tokenizer(text).input_ids[1:]
    for i in range(0, len(tokens), context_len):
        context = tokens[i:i+context_len]
        # print(context)
        target = tokens[i+context_len:i+context_len+completion_len]
        if i + context_len + completion_len >= len(tokens):
            break
        context_avg_len += len(context)
        target_avg_len += len(target)
        to_save.append({"input": tokenizer.decode(context), "output": tokenizer.decode(target)})

mode = mode_map[mode]
print(f"Context avg len: {context_avg_len/len(to_save)}, target avg len: {target_avg_len/len(to_save)}")
print("saving lines...")
with jsonlines.open(f"datasets/wikitext-103/{mode}.jsonl", "w") as f:
    f.write_all(to_save)