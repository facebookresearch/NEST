from rouge import Rouge
from collections import defaultdict

class WikiText103Task:
    def __init__(self, start_tag="", end_tag=""):
        self.metrics = ["rouge-1", "rouge-2", "rouge-l"]
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.template = "{start_tag} Write an article.\nArticle: {end_tag} {input}"
        self.rouge = Rouge()

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
        all_scores = self.rouge.get_scores(target, prediction)
        results = defaultdict(list)
        for scores in all_scores:
            for metric, score in scores.items():
                for sub_metric, sub_score in score.items():
                    if metric in self.metrics:
                        results[metric + f"-{sub_metric}"].append(sub_score)
        return results