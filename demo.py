###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
from models.knn_transformers import KNNTransformer
from generation import generate

# Initialize model
model = KNNTransformer(lm_model="meta-llama/Llama-2-7b-chat-hf",
                       retrieval_qry_model="facebook/dragon-plus-query-encoder",
                       corpus_path="corpora/enwiki-dec2021/text-list-100-sec.npy",
                       sparse_passage_dir="sparse_indexes/enwiki-dec2021/",
                       dense_passage_dir="dense_indexes/enwiki-dec2021/",
                       threads=32,
                       n_probe=4096, 
                       passage_k=20,
                       token_k=1024,
                       alpha=0.3,
                       tau=0.1,
                       window=64,
                       pad_id=-1,
                       pre_tokenized=True,
                       pre_retrieved=False,
                       fusion_coef=0.3)
# Retrieval
queries = ["Lanny Flaherty"]
query_inputs = model.passage_tokenizer(queries, return_tensors='pt', padding=True)
_, batch_doc_ids = model.retrieve_passage(queries, query_inputs)
model.build_token_index(batch_doc_ids)

prompt=["[INST] Question: Tell me a bio of Lanny Flaherty. Answer: [/INST]"]
tokens, doc_ids, poses = generate(model,
                                  prompt,
                                  max_prompt_len=128,
                                  max_gen_len=512,
                                  use_sampling=False,
                                  temp=1.0,
                                  top_k=0,
                                  top_p=0.0,
                                  beta=0.5,
                                  gamma=5e-4)
eos_id = model.tokenizer.eos_token_id
output = [model.tokenizer.decode((t+[eos_id])[:(t+[eos_id]).index(eos_id)]) for t in tokens]
print("Geneartion:", output)
print("Document ids (token-level):", doc_ids)