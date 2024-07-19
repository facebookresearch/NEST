import sys
import jsonlines

mode = sys.argv[1]
with open(f"datasets/triviaqa/triviaqa-{mode}.csv") as f:
    lines = f.readlines()

data = []
for line in lines:
    question, answers = line.strip("\n").split("\t")
    try:
        answers = eval(answers)
    except:
        print("skipping unparsable questions")
        continue
    data.append({"input":question.strip(), "output":answers})

print("saving lines...")
with jsonlines.open(f"datasets/triviaqa/{mode}.jsonl", "w") as f:
    f.write_all(data)