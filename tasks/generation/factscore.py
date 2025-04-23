###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
from rouge import Rouge
from collections import defaultdict

class FactScoreTask:
    def __init__(self, start_tag="", end_tag=""):
        self.metrics = ["rouge-1", "rouge-2", "rouge-l"]
        self.prompt_len = 128
        self.max_len = 512
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.template = "{start_tag} {input} Answer: {end_tag}"
        self.rouge = Rouge()

    def get_query(self, data):
        return data["topic"]

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
        all_scores = self.rouge.get_scores(target, prediction)
        results = defaultdict(list)
        for scores in all_scores:
            for metric, score in scores.items():
                for sub_metric, sub_score in score.items():
                    if metric in self.metrics:
                        results[metric + f"-{sub_metric}"].append(sub_score)
        return results