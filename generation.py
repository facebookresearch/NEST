###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
import torch
from typing import Optional, List, Tuple
from models.knn_transformers import KNNTransformer
from transformers import AutoTokenizer

@torch.inference_mode()
def generate(
    model: KNNTransformer,
    prompt: List[str],
    max_prompt_len: int = 256,
    max_gen_len: int = 256,
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    beta: float = 0.5, # threshold for dynamic selection
    gamma: float = 5e-2, # rejection leniency
) -> List[List[int]]:
    model.eval()
    pad_id = model.pad_id
    tokenizer = model.tokenizer
    prompt_tokens = [tokenizer(t)["input_ids"] for t in prompt]
    prompt_tokens = [
        t if len(t) < max_prompt_len else t[len(t) - max_prompt_len :]
        for t in prompt_tokens
    ]

    start_pos, end_pos = get_generation_range(prompt_tokens, max_gen_len)
    bsz = len(prompt_tokens)
    tokens = torch.full((bsz, end_pos), pad_id).cuda().long()
    doc_ids = torch.full((bsz, end_pos), pad_id).cuda().long()
    poses = torch.full((bsz, end_pos), pad_id).cuda().long()    
    # copy input tokens to tensor containing generated tokens
    for k, ex_tokens in enumerate(prompt_tokens):
        tokens[k, : len(ex_tokens)] = torch.tensor(ex_tokens).long()
    
    prev_pos = 0
    curr_pos = start_pos
    ngram_mask = None
    prev_cache = None
    while curr_pos < end_pos:
        logits, cache = model(tokens[:, prev_pos:curr_pos], cache=prev_cache, ngram_mask=ngram_mask)
        if prev_cache is None:
            prev_cache = cache

        knn_raw_scores = model.logs["knn_raw_scores"]
        knn_tokens = model.logs["knn_tokens"]
        knn_docids = model.logs["knn_docids"]
        knn_poses = model.logs["knn_poses"]
        knn_inter_coef = model.logs["knn_coef"]

        unmerged_probs = torch.softmax(logits, dim=-1)
        probs = unmerged_probs[:, :tokenizer.vocab_size] + unmerged_probs[:, tokenizer.vocab_size:]
        logits = torch.log(probs)
        
        if use_sampling:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = sample_top_p(probs, top_p)
            elif top_k > 0:
                next_token = sample_top_k(probs, top_k)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            next_token = next_token.reshape(-1)
        else:
            next_token = torch.argmax(logits, dim=-1)

        # n-gram replacement. If there're already n-grams in the target tokens, abstain
        next_token_group = torch.where(knn_inter_coef >= beta, 0, 1)
        
        # select the top-num_branch n-grams
        _, knn_topk_indices = torch.topk(knn_raw_scores, dim=-1, k=1)
        knn_topk_indices = knn_topk_indices.unsqueeze(-1).tile((1, 1, knn_tokens.shape[-1]))
        knn_tokens = torch.gather(knn_tokens, dim=1, index=knn_topk_indices)[:, 0]
        knn_docids = torch.gather(knn_docids, dim=1, index=knn_topk_indices)[:, 0]
        knn_poses = torch.gather(knn_poses, dim=1, index=knn_topk_indices)[:, 0]

        next_tokens = torch.cat([next_token.unsqueeze(-1), torch.ones_like(knn_tokens[:, 1:]) * pad_id], -1)
        next_docids = torch.ones_like(knn_docids) * pad_id
        next_poses = torch.ones_like(knn_poses) * pad_id
        next_docids[:, 0] = torch.where(next_token.unsqueeze(-1) == knn_tokens, knn_docids, next_docids)[:, 0]
        next_poses[:, 0] = torch.where(next_token.unsqueeze(-1) == knn_tokens, knn_poses, next_poses)[:, 0]
        
        # select n-gram or sampled token from knnlm
        next_token_mask = (next_token_group == 0) | (knn_tokens[:, :1] == pad_id)
        knn_tokens = torch.where(next_token_mask, next_tokens, knn_tokens)
        knn_docids = torch.where(next_token_mask, next_docids, knn_docids)
        knn_poses = torch.where(next_token_mask, next_poses, knn_poses)

        # remove DUMMY tokens
        max_span_length = (knn_tokens != pad_id).to(torch.int64).sum(-1).max()
        knn_tokens = knn_tokens[:, :max_span_length].contiguous()
        knn_docids = knn_docids[:, :max_span_length].contiguous()
        knn_poses = knn_poses[:, :max_span_length].contiguous()
        knn_end_pos = min(curr_pos + knn_tokens.shape[-1], end_pos)

        # there're already tokens predicted in the future slot, give up the current prediction
        mask = (tokens != pad_id)[:, curr_pos:knn_end_pos].to(torch.long).sum(-1, True) > 0
        knn_tokens = torch.where(mask, tokens[:, curr_pos:knn_end_pos], knn_tokens[:, :knn_end_pos-curr_pos])
        knn_docids = torch.where(mask, doc_ids[:, curr_pos:knn_end_pos], knn_docids[:, :knn_end_pos-curr_pos])
        knn_poses = torch.where(mask, poses[:, curr_pos:knn_end_pos], knn_poses[:, :knn_end_pos-curr_pos])

        # rejection sampling
        # take the current segment to be evaled by the lm model
        tokens_to_be_accepted = knn_tokens
        docids_to_be_accepted = knn_docids
        poses_to_be_accepted = knn_poses
        if max_span_length == 1 and next_token_group.sum() == 0:
            acception_logits = logits.unsqueeze(1)
        else:
            tbe = torch.where(tokens_to_be_accepted != pad_id, tokens_to_be_accepted, tokenizer.eos_token_id)
            tbe = torch.cat([tokens[:, curr_pos-1:curr_pos], tbe[:, :-1]], -1)
            prev_cache = [(c1[:, :, :-1], c2[:, :, :-1]) for c1, c2 in cache] if prev_pos == 0 else prev_cache
            acception_logits, cache = model(tbe, cache=prev_cache, mode="evaluation")

        # rejection sampling / decoding
        acception_dist = torch.softmax(acception_logits, dim=-1).view(-1, acception_logits.shape[-1])
        p = acception_dist[torch.arange(acception_dist.shape[0]), tokens_to_be_accepted.view(-1)]      
        baseline_p = 0.5 * gamma * acception_dist.max(-1).values
        p = p.view(acception_logits.shape[0], acception_logits.shape[1])
        p = torch.where(mask, 1, p) # accept tokens that have already been predicted in cur_pos:knn_end_pos
        p = torch.where(tokens_to_be_accepted == pad_id, 0, p) # reject dummy n-grams
        tokens_to_be_accepted_mask = torch.cumprod((baseline_p.view(*p.shape) < p).to(p.dtype), -1)
        
        # fill-in the accepted tokens in the generation
        tokens[:, curr_pos:knn_end_pos] = torch.where((tokens_to_be_accepted_mask > 0) | (next_token_group == 0), tokens_to_be_accepted, tokens[:, curr_pos:knn_end_pos])
        doc_ids[:, curr_pos:knn_end_pos] = torch.where((tokens_to_be_accepted_mask > 0) | (next_token_group == 0), docids_to_be_accepted, pad_id)
        poses[:, curr_pos:knn_end_pos] = torch.where((tokens_to_be_accepted_mask > 0) | (next_token_group == 0), poses_to_be_accepted, pad_id)
        move_forward = tokens_to_be_accepted_mask.to(torch.int64).sum(-1)
        acception_logits = acception_logits.view(-1, acception_logits.shape[-1])

        # resample the tokens that have been rejected
        if use_sampling:
            probs = torch.softmax(acception_logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = sample_top_p(probs, top_p)
            elif top_k > 0:
                next_token = sample_top_k(probs, top_k)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            next_token = next_token.reshape(-1, knn_end_pos - curr_pos)
        else:
            next_token = torch.argmax(acception_logits, dim=-1)
        next_token = next_token.view(tokens_to_be_accepted.shape[0], tokens_to_be_accepted.shape[1])
        
        for i in range(next_token.shape[0]):
            if move_forward[i] < knn_tokens.shape[-1]:
                j = curr_pos + move_forward[i]
                tokens[i, j] = torch.where((next_token_group[i, 0] != 0) & (tokens[i, j] == pad_id), next_token[i, move_forward[i]], tokens[i, j])
        move_forward = torch.where(tokens[:, curr_pos:knn_end_pos] != pad_id, 1, 0).sum(-1).min().detach().cpu().item()
        assert move_forward > 0
        model.mask_token_index(doc_ids[:, curr_pos:curr_pos + 1], poses[:, curr_pos:curr_pos + 1])
        
        # streaming: if the last token in the proposed n-gram is not rejected, we will fetch its continued segments in the passages instead of knnlm inference + sampling
        batch_idx = torch.arange(tokens.shape[0]).to(tokens.device)
        knn_ids = doc_ids[batch_idx, curr_pos + move_forward - 1] * model.token_index.max_len + poses[batch_idx, curr_pos + move_forward - 1]
        ngram_mask = torch.where((tokens_to_be_accepted_mask[batch_idx, move_forward - 1] > 0) & (doc_ids[batch_idx, curr_pos + move_forward - 1] != pad_id) & (poses[batch_idx, curr_pos + move_forward - 1] != pad_id), knn_ids, pad_id)
        # move forward and setup cache
        curr_pos += move_forward
        prev_pos = curr_pos - 1
        prev_cache = [(c1[:, :, :curr_pos], c2[:, :, :curr_pos]) for c1, c2 in cache]

    generated_tokens = [
        t[len(prompt_tokens[i]) : len(prompt_tokens[i]) + max_gen_len].tolist()
        for i, t in enumerate(tokens)
    ]    
    generated_doc_ids = [
        t[len(prompt_tokens[i]) : len(prompt_tokens[i]) + max_gen_len].tolist()
        for i, t in enumerate(doc_ids)
    ]
    generated_poses = [
        t[len(prompt_tokens[i]) : len(prompt_tokens[i]) + max_gen_len].tolist()
        for i, t in enumerate(poses)
    ]
    return (generated_tokens, generated_doc_ids, generated_poses)

def sample_top_k(probs, k):
    topk_value, _ = torch.topk(probs, k)  # batch_sz x topk
    min_value_top_k = topk_value[:, [-1]]
    probs[probs < min_value_top_k] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def get_min_length(input_tokens: Optional[List[List[int]]]) -> int:
    # reduce min length prompt over all processes to have an equal number of call on each process with fsdp
    if input_tokens is None:
        min_length = int(1e9)
    else:
        min_length = min([len(t) for t in input_tokens])
    return min_length


def get_max_length(input_tokens: Optional[List[List[int]]]) -> int:
    # reduce max length prompt over all processes to have an equal number of call on each process with fsdp
    if input_tokens is None:
        max_length = 0
    else:
        max_length = max([len(t) for t in input_tokens])
    return max_length


def get_generation_range(
    prompt_tokens: Optional[List[List[int]]], max_gen_len: int
) -> Tuple[int, int]:
    batch_min_prompt_length = get_min_length(prompt_tokens)
    batch_max_prompt_length = get_max_length(prompt_tokens)
    return batch_min_prompt_length, batch_max_prompt_length + max_gen_len