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

class MedMCQATask:
    def __init__(self, start_tag="", end_tag=""):
        self.metrics = ["alr"]
        self.prompt_len = 128
        self.max_len = 256
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.template = "{start_tag} Question: {input} Answer: {end_tag}"

    def get_query(self, data):
        return data["input"]

    def get_prompt(self, data):
        prompts = []
        for d in data["input"]:
            prompts.append(self.template.format(start_tag=self.start_tag, end_tag=self.end_tag, input=d).strip())
        return prompts

    def get_target(self, data):
        return data["output"]

    def get_support(self, data, k):
        return [support[:k] for support in data["support"]]
    
    def evaluate(self, prediction, target):
        results = defaultdict(list)
        for metric in self.metrics:
            if metric == "alr":
                for pred, ans in zip(prediction, target):
                    results["alr"].append(alr(pred, ans))
        return results