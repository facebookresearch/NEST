export CUDA_VISIBLE_DEVICES=3

python eval.py --mode test \
--output_dir results \
--data_dir datasets \
--tasks factscore \
--corpus_path corpora/enwiki-dec2021/text-list-100-sec.npy \
--sparse_passage_dir indexes/enwiki-dec2021/ \
--lm_model meta-llama/Llama-2-7b-chat-hf \
--retrieval_qry_model facebook/dragon-plus-query-encoder \
--retrieval_ctx_model facebook/dragon-plus-context-encoder \
--batch_size 2 \
--num_passages 10 \
--prompt_len 128 \
--max_len 128 \
--pre_tokenized \
--start_tag "[INST]" \
--end_tag "[/INST]" \
