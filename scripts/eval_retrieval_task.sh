###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
export CUDA_VISIBLE_DEVICES=7
# ,wikitext-103_retrieval,nq_retrieval,triviaqa_retrieval,medmcqa_retrieval,hotpotqa_retrieval,truthfulqa_retrieval
python eval.py --mode test \
--output_dir results \
--data_dir datasets \
--tasks medmcqa_retrieval,triviaqa_retrieval \
--corpus_path corpora/enwiki-dec2021/text-list-100-sec.jsonl \
--sparse_passage_dir sparse_indexes/enwiki-dec2021/ \
--dense_passage_dir dense_indexes/enwiki-dec2021/ \
--lm_model meta-llama/Llama-2-7b-chat-hf \
--retrieval_qry_model facebook/dragon-plus-query-encoder \
--batch_size 32 \
--num_passages 40 \
--start_tag "[INST]" \
--end_tag "[/INST]"
