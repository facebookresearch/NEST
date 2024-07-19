python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ../collections/enwiki-dec2021 \
  --index ../indexes/enwiki-dec2021 \
  --generator DefaultLuceneDocumentGenerator \
  --threads 32 \
  --optimize