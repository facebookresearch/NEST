###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################

import json
import numpy as np

def load_lines(path, tokenizer, prefix_len, target_len, batch_size=8):
    buffer = []
    n_toks = (1 + prefix_len + target_len) * batch_size
    with open(path, encoding="utf-8") as f:
        for line in f.readlines():
            buffer.extend(tokenizer(line.rstrip().replace("\n", ""))["input_ids"][1:])
            while len(buffer) >= n_toks:
                x = np.array(buffer[:n_toks]).reshape(batch_size, -1)
                buffer = buffer[n_toks:]
                assert x.shape[1] == 1 + prefix_len + target_len
                prefix = x[:, :prefix_len]
                y = x[:, prefix_len+1:prefix_len+target_len+1]
                x = x[:, prefix_len:prefix_len+target_len]
                yield dict(query=[tokenizer.decode(p) for p in prefix.tolist()], prefix=prefix, target=y, input=x[:, :y.shape[1]])

            
def load_datasets(path, batch_size=8):
    batch_inputs = []
    batch_outputs = []
    batch_topics = []
    batch_supports = []
    with open(path) as f:
        for line in f:
            if len(batch_inputs) == batch_size:
                yield dict(input=batch_inputs, output=batch_outputs, topic=batch_topics, support=batch_supports)
                batch_inputs, batch_outputs, batch_topics, batch_supports = [], [], [], []
            data = json.loads(line)
            batch_inputs.append(data["input"])
            batch_outputs.append(data["output"])
            batch_topics.append(data.get("topic", data["input"]))
            batch_supports.append(data.get("support", None))
    if len(batch_inputs) > 0:
        yield dict(input=batch_inputs, output=batch_outputs, topic=batch_topics, support=batch_supports)

def alr(pred, answers):
    has_answer = False
    pred = pred.lower().strip()
    for answer in answers:
        if answer.lower().strip() in pred:
            has_answer = True
        return int(has_answer)
    return int(has_answer)