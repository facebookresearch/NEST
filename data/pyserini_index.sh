###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input collections/enwiki-dec2021 \
  --index indexes/enwiki-dec2021 \
  --generator DefaultLuceneDocumentGenerator \
  --threads 32 \
  --optimize