###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
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