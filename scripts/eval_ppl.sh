###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
export CUDA_VISIBLE_DEVICES=3
MODEL_SIZE=7b
BATCH_SIZE=8

python eval.py --mode test \
--output_dir results/ \
--data_dir datasets \
--corpus_path corpora/enwiki-dec2021/text-list-100-sec.npy \
--sparse_passage_dir sparse_indexes/enwiki-dec2021/ \
--dense_passage_dir dense_indexes/enwiki-dec2021/ \
--lm_model meta-llama/Llama-2-${MODEL_SIZE}-chat-hf \
--retrieval_qry_model facebook/dragon-plus-query-encoder \
--batch_size $BATCH_SIZE \
--num_passages 40 \
--num_tokens 1024 \
--pre_tokenized \
--start_tag "[INST]" \
--end_tag "[/INST]" \
--ppl_files "datasets/wikitext-103/wiki.test.tokens" \
--n_batches 50 \
--prefix_len 128 \
--target_len 256
