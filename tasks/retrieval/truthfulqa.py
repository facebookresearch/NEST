###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
from utils import alr
from collections import defaultdict

def split_multi_answer(ans, sep=';', close=True):

    """Splits string of all reference answers into a list of formatted answers"""

    answers = ans.strip().split(sep)
    split_answers = []
    for a in answers:
        a = a.strip()
        if len(a):
            if close:  # add a period after all answers
                if a[-1] != '.':
                    split_answers.append(a + '.')
                else:
                    split_answers.append(a)
            else:
                split_answers.append(a)

    return split_answers

class TruthfulQARetrievalTask:
    def __init__(self):
        self.metrics = ["top-5", "top-20", "top-100"]
        self.max_len = 512
        
    def get_query(self, data):
        return data["input"]

    def get_target(self, data):
        return data["output"]

    def evaluate(self, prediction, target):
        results = defaultdict(list)
        for pred, tgt in zip(prediction, target):
            hit = False
            tgt = tgt["correct"]
            tgt = split_multi_answer(tgt)
            for i, p in enumerate(pred):
                p = p["title"] + " " + p["text"]
                if alr(p, tgt) > 0:
                    hit = True
                    break
            if hit:
                for metric in self.metrics:
                    k = int(metric.split("-")[-1]) - 1
                    if k >= i:
                        results[metric].append(1)
                    else:
                        results[metric].append(0)
            else:
                for metric in self.metrics:
                    results[metric].append(0)
        return results