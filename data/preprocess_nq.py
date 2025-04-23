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

mode = sys.argv[1]
with open(f"datasets/nq/nq-{mode}.csv") as f:
    lines = f.readlines()

data = []
for line in lines:
    question, answers = line.strip("\n").split("\t")
    answers = eval(answers)
    data.append({"input":question.strip(), "output":answers})

print("saving lines...")
with jsonlines.open(f"datasets/nq/{mode}.jsonl", "w") as f:
    f.write_all(data)