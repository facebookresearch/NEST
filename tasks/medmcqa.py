from utils import alr
from collections import defaultdict

class MedMCQATask:
    def __init__(self, start_tag="", end_tag=""):
        self.metrics = ["alr"]
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.template = "{start_tag} Question: {input}\n Answer: {end_tag}"

    def get_query(self, data):
        return data["input"]

    def get_prompt(self, data):
        prompts = []
        for d in data["input"]:
            prompts.append(self.template.format(start_tag=self.start_tag, end_tag=self.end_tag, input=d).strip())
        return prompts

    def get_target(self, data):
        return data["output"]

    def evaluate(self, prediction, target):
        results = defaultdict(list)
        for metric in self.metrics:
            if metric == "alr":
                for pred, ans in zip(prediction, target):
                    results["alr"].append(alr(pred, ans))
        return results