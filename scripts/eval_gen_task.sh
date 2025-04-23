###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
export CUDA_VISIBLE_DEVICES=3,4
MODEL_SIZE=7b
GAMMA=5e-4
BATCH_SIZE=4

python eval.py --mode test \
--output_dir results/ \
--data_dir datasets \
--tasks factscore \
--corpus_path corpora/enwiki-dec2021/text-list-100-sec.npy \
--sparse_passage_dir sparse_indexes/enwiki-dec2021/ \
--dense_passage_dir dense_indexes/enwiki-dec2021/ \
--lm_model meta-llama/Llama-2-${MODEL_SIZE}-chat-hf \
--retrieval_qry_model facebook/dragon-plus-query-encoder \
--batch_size $BATCH_SIZE \
--num_passages 40 \
--pre_tokenized \
--pre_retrieved \
--start_tag "[INST]" \
--end_tag "[/INST]" \
--gamma $GAMMA
