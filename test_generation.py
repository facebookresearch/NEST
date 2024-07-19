from knn_transformers import KNNTransformer
from generation import generate

# Initialize model
model = KNNTransformer(lm_model="Trelis/Llama-2-7b-chat-hf-hosted-inference-8bit",
                       retrieval_qry_model="facebook/dragon-plus-query-encoder",
                       retrieval_ctx_model="facebook/dragon-plus-context-encoder",
                       corpus_path="corpora/enwiki-dec2021/text-list-100-sec.npy",
                       sparse_passage_dir="indexes/enwiki-dec2021/",
                       dense_passage_dir=None,
                       threads=32,
                       n_probe=4096, 
                       passage_k=10,
                       token_k=512,
                       alpha=0.3,
                       tau=0.1,
                       window=64,
                       pad_id=-1,
                       pre_tokenized=True,
                       pre_retrieved=False,
                       fusion_coef=0.3)
# Retrieval
queries = ["Lanny Flaherty", "Taral Hicks"]
query_inputs = model.passage_tokenizer(queries, return_tensors='pt', padding=True)
_, batch_doc_ids = model.retrieve_passage(queries, query_inputs)
model.build_token_index(batch_doc_ids)

prompt=["[INST] Question: Tell me a bio of Lanny Falherty.\nAnswer: [/INST]", "[INST] Question: Tell me a bio of Taral Hicks.\nAnswer: [/INST]"]
tokens, doc_ids, poses = generate(model,
                                  prompt,
                                  max_prompt_len=256,
                                  max_gen_len=256,
                                  use_sampling=False,
                                  temp=1.0,
                                  top_k=0,
                                  top_p=0.0,
                                  beta=0.5,
                                  gamma=5e-4)
output = [model.tokenizer.decode(t) for t in tokens]
print(output)