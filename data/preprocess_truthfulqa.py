import sys
import jsonlines
import pandas as pd

data = pd.read_csv(f"datasets/truthfulqa/TruthfulQA.csv")
data.dropna(axis=1, how='all', inplace=True)  # drop all-null columns

results = []
for idx in data.index:
    results.append({"input":data.loc[idx, "Question"], "output": {"correct":data.loc[idx, "Correct Answers"], "incorrect":data.loc[idx, "Incorrect Answers"]}})

print("saving lines...")
with jsonlines.open(f"datasets/truthfulqa/test.jsonl", "w") as f:
    f.write_all(results)