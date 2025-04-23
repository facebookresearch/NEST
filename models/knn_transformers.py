###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
from typing import Optional, Tuple, List, Callable, Dict, Any
from tqdm import tqdm
import json
import faiss
import numpy as np

import torch
from torch.nn import functional as F
import torch_scatter

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from pyserini.search.lucene import LuceneSearcher
from multiprocessing import Pool

class TokenIndex(object):
    def __init__(self, pad_id=-1, window=1, sub_batch_size=10):
        self.window = window
        self.pad_id = pad_id
        self.sub_batch_size = sub_batch_size
    
    def build_index(self, model, doc_ids, tokens, token_poses):
        bsz, passage_k = doc_ids.shape
        max_len = tokens.shape[-1] - 1
        knn_ids = doc_ids.view(-1, 1) * max_len + (token_poses[:, 1:] - 1)
        vals = tokens[:, 1:]

        token_mask = (token_poses[:, :-1] != self.pad_id) & (doc_ids.view(-1, 1) != -1)
        unmasked_tokens = torch.where(tokens[:, :-1] != self.pad_id, tokens[:, :-1], 0)    
        effective_tok_embeddings = []
        for i in range(0, unmasked_tokens.shape[0], self.sub_batch_size):
            outputs = model(unmasked_tokens[i:i+self.sub_batch_size], output_hidden_states=True)
            tok_embeddings = outputs.hidden_states[-2]
            tok_embeddings = tok_embeddings.contiguous().view(-1, tok_embeddings.shape[-1])
            effective_tok_embeddings.append(tok_embeddings[token_mask[i:i+self.sub_batch_size].reshape((-1,))])
        effective_tok_embeddings = torch.cat(effective_tok_embeddings, 0)
        token_mask = token_mask.reshape((-1,))

        self.tok_embeddings = torch.zeros(bsz * max_len * passage_k, effective_tok_embeddings.shape[-1]).cuda().to(torch.bfloat16)
        self.tok_square = torch.zeros(bsz * max_len * passage_k).cuda().to(torch.bfloat16)
        self.tok_mask = torch.zeros(bsz * max_len * passage_k).cuda().to(torch.int32)

        self.tok_embeddings[token_mask] = effective_tok_embeddings
        self.tok_square[token_mask] = (effective_tok_embeddings.float()**2).sum(1).to(torch.bfloat16)
        self.tok_mask[token_mask] = 1

        self.tok_embeddings = torch.permute(self.tok_embeddings.reshape((bsz, max_len * passage_k, effective_tok_embeddings.shape[-1])), (0, 2, 1))
        self.tok_square = self.tok_square.reshape((bsz, 1, max_len * passage_k))
        self.tok_mask = self.tok_mask.reshape((bsz, 1, max_len * passage_k))
        
        self.knn_ids = knn_ids.reshape((bsz, 1, max_len * passage_k))
        self.vals = vals.reshape((bsz, 1, max_len * passage_k))
        self.max_len = max_len
        self.passage_k = passage_k

    def search(self, query_embeddings, k, ngram_mask=None):
        if ngram_mask is not None:
            mask = (ngram_mask != self.pad_id)
            tok_square = self.tok_square[~mask]
            tok_embeddings = self.tok_embeddings[~mask]
            tok_mask = self.tok_mask[~mask]
            vals = self.vals[~mask]
            knn_ids = self.knn_ids[~mask]
        else:
            tok_square = self.tok_square
            tok_embeddings = self.tok_embeddings
            tok_mask = self.tok_mask
            vals = self.vals
            knn_ids = self.knn_ids
        distance = tok_square + (query_embeddings.float()**2).sum(-1, True)
        distance = distance - 2 * torch.bmm(query_embeddings.float(), tok_embeddings.float())
        distance = torch.where(tok_mask > 0, distance, float('inf'))
        distance, idx = torch.topk(distance, k=k, dim=-1, largest=False) # B x T x K
        tokens, doc_ids, token_poses = self.get_n_gram_by_idx(idx, self.window, vals, knn_ids)
        return -distance.view(-1, k), tokens.view(-1, k, self.window), doc_ids.view(-1, k, self.window), token_poses.view(-1, k, self.window)

    def get_n_gram_by_idx(self, idx, window, vals, knn_ids):
        idx = idx.unsqueeze(-1) + torch.arange(window).to(idx.dtype).to(idx.device).view(1, 1, 1, window) # B x T x K x N
        idx = idx.view(idx.shape[0], idx.shape[1], -1)
        
        vals = torch.cat([vals, torch.ones_like(vals[:, :, :window]) * self.pad_id], -1)
        knn_ids = torch.cat([knn_ids, torch.ones_like(knn_ids[:, :, :window]) * self.pad_id], -1)
        
        tokens = torch.take_along_dim(vals, idx, dim=-1).view(idx.shape[0], idx.shape[1], -1, window) # B x T x K x N
        knn_ids = torch.take_along_dim(knn_ids, idx, dim=-1).view(idx.shape[0], idx.shape[1], -1, window)
        doc_ids, token_poses = knn_ids // self.max_len, knn_ids % self.max_len
        
        mask = token_poses[:, :, :, 1:] > token_poses[:, :, :, :-1] # to ensure we won't retrieve tokens from other passages if token pose is near the end
        tokens[:, :, :, 1:] = torch.where(mask, tokens[:, :, :, 1:], self.pad_id)
        doc_ids[:, :, :, 1:] = torch.where(mask, doc_ids[:, :, :, 1:], self.pad_id)
        token_poses[:, :, :, 1:] = torch.where(mask, token_poses[:, :, :, 1:], self.pad_id)
        return tokens, doc_ids, token_poses

    def get_n_gram_by_mask(self, mask):
        knn_ids = mask[mask != self.pad_id].unsqueeze(-1) # B' x 1
        total_knn_ids = self.knn_ids[mask != self.pad_id].squeeze(1) # B' x (max_len * passage_k)
        idx = torch.where(knn_ids == total_knn_ids)
        out = torch.ones_like(knn_ids).view(-1) * torch.max(idx[1])
        out.scatter_reduce(0, idx[0], idx[1], reduce="amin")
        idx = out.view(knn_ids.shape[0], 1, 1)
        tokens, doc_ids, token_poses = self.get_n_gram_by_idx(idx, self.window + 1, self.vals[torch.where(mask != self.pad_id)], self.knn_ids[torch.where(mask != self.pad_id)])
        tokens, doc_ids, token_poses = tokens[:, :, :, 1:], doc_ids[:, :, :, 1:], token_poses[:, :, :, 1:]
        return tokens.view(-1, 1, self.window), doc_ids.view(-1, 1, self.window), token_poses.view(-1, 1, self.window)

    def mask_vectors(self, doc_ids, token_poses):
        knn_ids = doc_ids * self.max_len + token_poses
        knn_ids = torch.where(token_poses > 0, knn_ids, self.pad_id)
        mask = (self.knn_ids.unsqueeze(-1) == knn_ids.unsqueeze(1).unsqueeze(-2)).to(torch.int64).sum(-1) > 0
        self.tok_mask = torch.where(mask, 0, self.tok_mask)


class KNNTransformer(torch.nn.Module):
    def __init__(self, lm_model: str = "meta-llama/Llama-2-7b-chat-hf",
                       retrieval_qry_model: str = "facebook/dragon-plus-query-encoder",
                       corpus_path: str = None, # data store
                       sparse_passage_dir: str = None, # sparse index path
                       dense_passage_dir: str = None, # dense index path
                       dense_rerank: bool = True, # whether to rerank on sparse
                       threads: int = 32, # threads for sparse/dense search
                       n_probe: int = 4096,  # probs for dense ivf index search
                       passage_k: int = 10, # top-k passages
                       token_k: int = 1024, # top-k tokens
                       alpha: float = 0.1, # RRC offset
                       tau: float = 0.1, # RRC scale
                       temp: float = 0.7, # LM temp
                       window: int = 1, # span length for proposal, can be increased and shrinked dynamically
                       pad_id: int = -1, # padding token
                       pre_tokenized: bool = True, # corpus pre-tokenized?
                       pre_retrieved: bool = False, # passages pre-retrieved?
                       fusion_coef: float = 0.3 # dense-sparse fusion weight
                       ):
        super().__init__()
        self.lm_model = AutoModelForCausalLM.from_pretrained(lm_model, torch_dtype=torch.bfloat16, 
                                                             attn_implementation="flash_attention_2",
                                                             device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model)
        self.passage_qry_model = AutoModel.from_pretrained(retrieval_qry_model, torch_dtype=torch.bfloat16).cuda()
        self.passage_tokenizer = AutoTokenizer.from_pretrained(retrieval_qry_model)
        # Passage-level index
        self.threads = threads
        self.pre_tokenized = pre_tokenized
        self.pre_retrieved = pre_retrieved
        self.dense_rerank = dense_rerank
        if corpus_path is not None:
            if pre_tokenized:
                print("Loading knn token corpus...")
                self.corpus_tokens = np.load(corpus_path, mmap_mode='r+')
            else:
                print("Loading corpus...")
                with open(corpus_path, "r", encoding="utf-8") as f, Pool(threads) as p:
                    self.corpus = list(tqdm(p.imap(json.loads, f, 4000), total=33176581))

        if not pre_retrieved:
            print("Loading dense passage index...")
            faiss.omp_set_num_threads(threads)
            self.dense_passage_index = faiss.read_index("/".join([dense_passage_dir, "passage.ivf65536.pq256.index"]))    
            faiss.ParameterSpace().set_index_parameter(self.dense_passage_index, "nprobe", n_probe)
            
            print("Loading sparse passage index...")
            self.sparse_passage_index = LuceneSearcher(sparse_passage_dir)
            
        # Index params
        self.token_k = token_k
        self.passage_k = passage_k
        self.window = window
        self.pad_id = pad_id
        # Interpolation
        self.alpha = alpha
        self.tau = tau
        self.temp = temp
        self.fusion_coef = fusion_coef

    def fusion(self, batch_dense_sim, batch_dense_doc_ids, batch_sparse_sim, batch_sparse_doc_ids, k, fusion_coef):
        if isinstance(fusion_coef, float):
            fusion_coef = fusion_coef * np.ones_like(batch_dense_sim[:, 0])
        batch_dense = [{doc_id:sim for doc_id, sim in zip(doc_ids, sims)} for doc_ids, sims in zip(batch_dense_doc_ids, batch_dense_sim)]
        batch_sparse = [{doc_id:sim for doc_id, sim in zip(doc_ids, sims)} for doc_ids, sims in zip(batch_sparse_doc_ids, batch_sparse_sim)]
        batch_min_dense_sim = batch_dense_sim[:, -1]
        batch_min_sparse_sim = batch_sparse_sim[:, -1]
        batch_fusion_doc_ids = []
        batch_fusion_doc_scores = []
        for i in range(len(batch_dense)):
            fusion_res = {}
            union_doc_ids = np.union1d(batch_dense_doc_ids[i], batch_sparse_doc_ids[i])
            for doc_id in union_doc_ids:
                score = fusion_coef[i] * batch_dense[i].get(doc_id, batch_min_dense_sim[i]) + (1 - fusion_coef[i]) * batch_sparse[i].get(doc_id, batch_min_sparse_sim[i])
                fusion_res[doc_id] = score
            fusion_doc_ids, fusion_doc_scores = list(zip(*(sorted([(k, v) for k, v in fusion_res.items()], key=lambda x: -x[1])[:k])))
            batch_fusion_doc_ids.append(fusion_doc_ids)
            batch_fusion_doc_scores.append(fusion_doc_scores)
        return np.array(batch_fusion_doc_ids), np.array(batch_fusion_doc_scores)

    def retrieve_passage(self, queries, query_inputs):
        sub_n = 4000
        batch_hits = self.sparse_passage_index.batch_search(queries, [str(i) for i in range(len(queries))], k=sub_n, threads=self.threads)
        sparse_doc_ids = np.zeros((len(queries), sub_n), dtype=np.int64)
        sparse_doc_scores = np.zeros((len(queries), sub_n), dtype=np.float32)
        for i, hits in batch_hits.items():
            for j, hit in enumerate(hits):
                sparse_doc_ids[int(i)][j] = hit.docid
                sparse_doc_scores[int(i)][j] = hit.score

        query_inputs = {k:v.cuda() for k, v in query_inputs.items()}
        query_embeddings = self.passage_qry_model(**query_inputs).last_hidden_state[:, 0, :].detach().float().cpu().numpy()
        dense_doc_scores, dense_doc_ids = self.dense_passage_index.search(query_embeddings, k=sub_n)
        
        sparse_coef = (1 - sparse_doc_scores[:, 100] / sparse_doc_scores.max(-1))
        dense_coef = (1 - np.where(dense_doc_scores[:, 100] > 0, dense_doc_scores[:, 100], 0) / dense_doc_scores.max(-1))
        fusion_coef = (1 - self.fusion_coef) * (1 - sparse_coef) + self.fusion_coef * dense_coef
        doc_ids, doc_scores = self.fusion(dense_doc_scores, dense_doc_ids, sparse_doc_scores, sparse_doc_ids, k=self.passage_k, fusion_coef=fusion_coef)
        return doc_scores, doc_ids

    def compute_inter_coef(self, knn_scores):
        coef = torch.sigmoid((knn_scores.max(-1, True).values / knn_scores.min(-1, True).values  - self.alpha) / self.tau)
        return coef

    @torch.no_grad()
    def forward(self, 
        tokens: torch.Tensor,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        mode: str = "generation",
        ngram_mask: torch.Tensor = None,
        eval_mask: torch.Tensor = None,
        ):
        self.logs = {} 
        # LM prediction
        outputs = self.lm_model(tokens, output_hidden_states=True, past_key_values=cache)
        lm_logits = outputs.logits
        cache = outputs.past_key_values
        h = outputs.hidden_states[-2]

        if mode == "generation":
            mask = (ngram_mask != self.pad_id) if ngram_mask is not None else None
            num_forward = (~mask).to(torch.int64).sum().detach().cpu().item() if ngram_mask is not None else 1 # number of samples that needs lm inference
            num_static = tokens.shape[0] - num_forward # number of samples whose tokens can be directly fetched from retrieved passages
            h = h[:, -1:]
            lm_logits = lm_logits[:, -1]
            if num_forward > 0 and ngram_mask is not None:
                h = h[torch.where(~mask)]
                lm_logits = lm_logits[~mask]
        elif mode == "evaluation":
            if eval_mask is not None:
                zeros = (~eval_mask).to(torch.int64).sum(-1).min()
                h = h[:, zeros:]

        lm_probs = F.softmax(lm_logits/self.temp, dim=-1)
        
        if mode == "evaluation" or mask is None or num_forward > 0:
            # KNN search
            raw_knn_scores, vals, doc_ids, poses = self.token_index.search(h.contiguous(), self.token_k, ngram_mask)
            # Gather knn prob
            knn_raw_probs = F.softmax(raw_knn_scores.to(torch.float32) / (h.shape[-1]**0.5), dim=-1)
            out = torch.zeros(vals.shape[0], lm_logits.shape[-1]).to(knn_raw_probs.device).to(knn_raw_probs.dtype)
            knn_probs = torch_scatter.scatter_add(src=knn_raw_probs, index=torch.where(vals == -1, 0, vals)[:, :, 0], out=out)
            knn_probs[:, 0] = 0
            # Interpolation
            inter_coef = self.compute_inter_coef(raw_knn_scores).to(lm_logits.dtype)
        else:
            inter_coef = 1.0
            knn_probs = 0.0

        if mode == "generation":
            if ngram_mask is None or num_forward > 0:
                probs = torch.cat([inter_coef * lm_probs, (1 - inter_coef) * knn_probs], -1)
                knn_mixed_probs = torch.gather(probs, index=torch.where(vals == -1, 0, vals)[:, :, 0], dim=-1)
                knn_probs[:, 0] = 0
                knn_raw_scores = knn_mixed_probs + 1e-6 * knn_raw_probs # global + eps * local
                logits = torch.log(probs)

            if ngram_mask is not None and num_static > 0:
                static_vals, static_docids, static_poses = self.token_index.get_n_gram_by_mask(ngram_mask) # B' x 1 x W
                total_vals = torch.zeros_like(static_vals[:1]).tile((tokens.shape[0], self.token_k, 1))
                total_docids = torch.zeros_like(static_docids[:1]).tile((tokens.shape[0], self.token_k, 1))
                total_poses = torch.zeros_like(static_poses[:1]).tile((tokens.shape[0], self.token_k, 1))
                total_knn_raw_scores = torch.zeros(tokens.shape[0], self.token_k).to(static_vals.device)
                total_logits = torch.zeros(tokens.shape[0], self.tokenizer.vocab_size * 2).to(static_vals.device)
                total_inter_coef = torch.zeros(tokens.shape[0], 1).to(static_vals.device)
                
                total_vals[mask] = static_vals
                total_docids[mask] = static_docids
                total_poses[mask] = static_poses
                tmp_logits = total_logits[mask]
                tmp_logits[torch.arange(num_static).to(static_vals.device), torch.where(static_vals == -1, 0, static_vals)[:, 0, 0] + self.tokenizer.vocab_size] = 1
                tmp_logits[:, 0] = 0
                total_logits[mask] = tmp_logits
                if num_forward > 0:
                    total_vals[~mask] = vals
                    total_docids[~mask] = doc_ids
                    total_poses[~mask] = poses
                    total_knn_raw_scores[~mask] = knn_raw_scores
                    total_logits[~mask] = logits
                    total_inter_coef[~mask] = inter_coef

                logits = total_logits
                self.logs["knn_raw_scores"] = total_knn_raw_scores
                self.logs["knn_tokens"] = total_vals
                self.logs["knn_docids"] = total_docids
                self.logs["knn_poses"] = total_poses
                self.logs["knn_coef"] = total_inter_coef
            else:
                self.logs["knn_raw_scores"] = knn_raw_scores
                self.logs["knn_tokens"] = vals
                self.logs["knn_docids"] = doc_ids
                self.logs["knn_poses"] = poses
                self.logs["knn_coef"] = inter_coef

        elif mode == "evaluation":
            if eval_mask is None:
                inter_coef = inter_coef.view(lm_probs.shape[0], lm_probs.shape[1], 1)
                knn_probs = knn_probs.view(lm_probs.shape[0], lm_probs.shape[1], -1)   
            else:    
                inter_coef = inter_coef.view(lm_probs.shape[0], eval_mask.shape[1] - zeros, 1)
                knn_probs = knn_probs.view(lm_probs.shape[0], eval_mask.shape[1] - zeros, -1)
                inter_coef = torch.cat([torch.ones_like(eval_mask[:, :zeros].to(inter_coef.dtype)).unsqueeze(-1), inter_coef], 1)
                knn_probs = torch.cat([torch.zeros_like(eval_mask[:, :zeros].to(knn_probs.dtype)).unsqueeze(-1).tile((1, 1, knn_probs.shape[-1])), knn_probs], 1)
            probs = inter_coef * lm_probs + (1 - inter_coef) * knn_probs
            logits = torch.log(probs)
        return logits, cache

    def build_token_index(self, batch_doc_ids):
        bsz, passage_k = batch_doc_ids.shape
        self.token_index = TokenIndex(self.pad_id, self.window)
        if not self.pre_tokenized:
            batch_docs = [[self.corpus[doc_id] for doc_id in doc_ids] for doc_ids in batch_doc_ids]
            tokens, token_poses = self.encode(batch_docs)
            doc_ids = batch_doc_ids
        else:
            doc_ids = batch_doc_ids.reshape((-1,))
            tokens = self.corpus_tokens[doc_ids].astype(np.int64)
            token_poses = np.tile(np.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1))
            token_poses = np.where(tokens != self.pad_id, token_poses, self.pad_id)
            doc_ids = doc_ids.reshape((bsz, passage_k))
        
        doc_ids = torch.from_numpy(doc_ids).cuda()
        tokens = torch.from_numpy(tokens).cuda()
        token_poses = torch.from_numpy(token_poses).cuda()
        self.token_index.build_index(self.lm_model, doc_ids, tokens, token_poses)
    
    def mask_token_index(self, doc_ids, token_poses):
        self.token_index.mask_vectors(doc_ids, token_poses)
    
    def encode(self, k_supports, max_len=-1):
        tokens, d_lens = [], []
        for supports in k_supports:
            for entry in supports:
                title, d = entry['title'], entry['text']
                d_token = self.tokenizer(f"{title} {d}")
                tokens.append(d_token)
                d_lens.append(len(d_token))
        max_len = max([len(x) for x in tokens])
        tokens = np.array([token + [self.pad_id] * (max_len - len(token)) for token in tokens])
        token_poses = np.array([[i for i in range(d_len)] + [self.pad_id] * (max_len - d_len) for d_len in d_lens])
        return tokens, token_poses