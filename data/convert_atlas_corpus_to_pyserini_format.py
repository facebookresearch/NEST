###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
import json
import jsonlines
import sys
from multiprocessing import Pool
from tqdm import tqdm

src_path = sys.argv[1]
dst_path = sys.argv[2]

results = []
with open(src_path, "r") as f, Pool(48) as p:
    print("Loading corpus...")
    with open(src_path, "r", encoding="utf-8") as f, Pool(32) as p:
        corpus = tqdm(list(p.imap(json.loads, f, 4000)), total=33176581)
        for data in tqdm(corpus):
            results.append({"id": data["id"], "contents": " ".join([data["title"], data["text"]])})

with jsonlines.open(dst_path, 'w') as writer:
    writer.write_all(results)