###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
from tasks import WikiText103Task, FactScoreTask, NQTask, TriviaQATask, MedMCQATask, HotPotQATask, TruthfulQATask
from tasks import WikiText103RetrievalTask, FactScoreRetrievalTask, NQRetrievalTask, TriviaQARetrievalTask, MedMCQARetrievalTask, HotPotQARetrievalTask, TruthfulQARetrievalTask
from models.knn_transformers import KNNTransformer
from generation import generate
from utils import load_datasets, load_lines

from collections import defaultdict
import os
import argparse
import json
import jsonlines
import torch
import torch.nn.functional as F
import math
import numpy as np

GEN_TASK = {
    "wikitext-103":WikiText103Task,
    "factscore":FactScoreTask,
    "nq":NQTask,
    "triviaqa":TriviaQATask,
    "medmcqa":MedMCQATask,
    "hotpotqa":HotPotQATask,
    "truthfulqa":TruthfulQATask,
}

RET_TASK = {
    "wikitext-103_retrieval":WikiText103RetrievalTask,
    "factscore_retrieval":FactScoreRetrievalTask,
    "nq_retrieval":NQRetrievalTask,
    "triviaqa_retrieval":TriviaQARetrievalTask,
    "medmcqa_retrieval":MedMCQARetrievalTask,
    "hotpotqa_retrieval":HotPotQARetrievalTask,
    "truthfulqa_retrieval":TruthfulQARetrievalTask,
}


@torch.no_grad()
def eval_ppl(args, model, path):
    def prompt(tokenizer, input_tokens, output_tokens):
        bx, by = [], []
        prompt = "Complete the following paragraph: "
        prompt_tokens = tokenizer(prompt)["input_ids"]
        for input_token, output_token in zip(input_tokens, output_tokens):
            x = prompt_tokens + input_token.tolist()
            y = [-100] * len(prompt_tokens) + output_token.tolist()
            bx.append(x)
            by.append(y)
        return np.array(bx), np.array(by)

    model.eval()
    
    data_iterator = load_lines(
        path=path,
        tokenizer=model.tokenizer,
        prefix_len=args.prefix_len,
        target_len=args.target_len,
        batch_size=args.batch_size,
    )
    metric = 0.0
    n_toks = 0

    for i, batch in enumerate(data_iterator):
        if i >= args.n_batches:
            break
        query = batch["query"]
        batch_inputs = model.passage_tokenizer(query, padding=True, return_tensors='pt', max_length=128, truncation=True)
        _, doc_ids = model.retrieve_passage(query, batch_inputs)
        model.build_token_index(doc_ids)

        prefix = batch["prefix"]
        y_prefix = -100 * np.ones_like(prefix)
        x = np.concatenate([prefix, batch["input"]], -1)
        y = np.concatenate([y_prefix, batch["target"]], -1)
        x, y = prompt(model.tokenizer, x, y)
        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()
        pred, _ = model(x, mode="evaluation", eval_mask=(y != -100))
        loss = F.cross_entropy(pred.flatten(0, 1), y.flatten(0, 1), reduction="sum")
        metric += loss.item()
        n_toks += (y != -100).to(torch.int64).sum().item()
        ppl = math.exp(metric / n_toks) if n_toks > 0 else 0.0
        print("PPL:", ppl)
    return ppl


@torch.no_grad()
def eval_retrieval(args, model, dataset, task_str):
    task = RET_TASK[task_str]()
    results = defaultdict(list)
    ret = []
    for i, data in enumerate(dataset):
        print(f"Batch {i}")
        queries = task.get_query(data)
        query_inputs = model.passage_tokenizer(queries, return_tensors='pt', padding=True)
        _, batch_doc_ids = model.retrieve_passage(queries, query_inputs)
        batch_docs = [[model.corpus[doc_id] for doc_id in doc_ids] for doc_ids in batch_doc_ids]
        metrics = task.evaluate(batch_docs, task.get_target(data))

        for key, val in metrics.items():
            results[key].extend(val)

        for it, ot, qt, doc_ids in zip(data["input"], data["output"], data["topic"], batch_doc_ids):
            if "factscore" in task_str:
                ret.append({"input": it, "output": ot, "topic": qt, "support":doc_ids.tolist()})
            else:
                ret.append({"input": it, "output": ot, "support":doc_ids.tolist()})
    results = {k: sum(v)/len(v) for k, v in results.items()}
    return results, ret


@torch.no_grad()
def eval_generation(args, model, dataset, task_str):
    task = GEN_TASK[task_str](start_tag=args.start_tag, end_tag=args.end_tag)
    results = defaultdict(list)
    outputs = []
    for i, data in enumerate(dataset):
        if i < 1:
            continue
        print(f"Batch {i}")
        queries = task.get_query(data)
        if args.pre_retrieved:
            batch_doc_ids = task.get_support(data, k=args.num_passages)
            batch_doc_ids = np.array(batch_doc_ids)
        else:
            query_inputs = model.passage_tokenizer(queries, return_tensors='pt', padding=True)
            _, batch_doc_ids = model.retrieve_passage(queries, query_inputs)
        model.build_token_index(batch_doc_ids)
        prompt = task.get_prompt(data)
        tokens, _, _ = generate(model,
                                prompt,
                                max_prompt_len=task.prompt_len,
                                max_gen_len=task.max_len,
                                use_sampling=False,
                                beta=args.beta,
                                gamma=args.gamma,
                                temp=1.0,
                                top_k=0,
                                top_p=0.0,)
        output = [model.tokenizer.decode(t) for t in tokens]
        metrics = task.evaluate(output, task.get_target(data))
        for key, val in metrics.items():
            results[key].extend(val)
        for it, ot, qt in zip(prompt, output, queries):
            if task_str == "factscore":
                outputs.append({"input":it, "output":ot, "topic":qt})
            else:
                outputs.append({"input":it, "output":ot})
    results = {k: sum(v)/len(v) for k, v in results.items()}
    return results, outputs


def task_evaluation(args):
    model = KNNTransformer(lm_model=args.lm_model,
            retrieval_qry_model=args.retrieval_qry_model,
            corpus_path=args.corpus_path,
            sparse_passage_dir=args.sparse_passage_dir,
            dense_passage_dir=args.dense_passage_dir,
            passage_k=args.num_passages,
            token_k=args.num_tokens,
            alpha=args.alpha,
            tau=args.tau,
            fusion_coef=args.fusion_coef,
            pre_tokenized=args.pre_tokenized,
            pre_retrieved=args.pre_retrieved,
            window=args.window,
            threads=args.threads,
            n_probe=args.nprobe, 
            pad_id=args.pad_id)

    if args.ppl_files is None:
        print("No ppl files to eval")
        ppl_files = []
    else:
        ppl_files = args.ppl_files.split(",")

    for ppl_file in ppl_files:
        print(f"Evaluating ppl for {ppl_file}...")
        task_str = ppl_file.split("/")[-1].split(".")[0]
        output_dir = "/".join([args.output_dir, "ppl", task_str])
        os.makedirs(output_dir, exist_ok=True)
        metrics = eval_ppl(args, model, ppl_file)
        print(f"Saving metrics for {task_str}...")
        with open("/".join([output_dir, "metrics.json"]), "w") as f:
            json.dump(metrics, f, indent=4)
    
    if args.tasks is None:
        print("No generation tasks to eval!")
        tasks = []
    else:
        tasks = args.tasks.split(",")
    
    print("Evaluating generation tasks...")
    for task_str in tasks:
        if task_str not in GEN_TASK:
            continue
        print(f"Evaluating {task_str}...")
        output_dir = "/".join([args.output_dir, "generation", task_str])
        os.makedirs(output_dir, exist_ok=True)
        dataset = load_datasets("/".join([args.data_dir, task_str, f"{args.mode}.jsonl"]), batch_size=args.batch_size)
        metrics, outputs = eval_generation(args, model, dataset, task_str)
        print(f"Saving outputs for {task_str}...")
        with jsonlines.open("/".join([output_dir, "results.jsonl"]), "w") as f:
            f.write_all(outputs)
        print(f"Saving metrics for {task_str}...")
        with open("/".join([output_dir, "metrics.json"]), "w") as f:
            json.dump(metrics, f, indent=4)
    
    
    print("Evaluating retrieval tasks...")
    model.lm_model.to('cpu')
    for task_str in tasks:
        if task_str not in RET_TASK:
            continue
        print(f"Evaluating {task_str}...")
        output_dir = "/".join([args.output_dir, "retrieval", task_str])
        os.makedirs(output_dir, exist_ok=True)
        dataset = load_datasets("/".join([args.data_dir, task_str.split("_")[0], f"{args.mode}.jsonl"]), batch_size=args.batch_size)
        metrics, outputs = eval_retrieval(args, model, dataset, task_str)
        print(f"Saving outputs for {task_str}...")
        with jsonlines.open("/".join([output_dir, "results.jsonl"]), "w") as f:
            f.write_all(outputs)
        print(f"Saving metrics for {task_str}...")
        with open("/".join([output_dir, "metrics.json"]), "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Task Evaluation')
    parser.add_argument('--mode', default="test", type=str,
                        help='Test or validation')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='Output path to save metrics')
    parser.add_argument('--data_dir', default="datasets/", type=str,
                        help='Path to data')
    parser.add_argument('--ppl_files', default=None, type=str,
                        help='Path to ppl files')
    parser.add_argument('--tasks', default=None, type=str,
                        help='Path to task')
    parser.add_argument('--corpus_path', default="corpora/enwiki-dec2021/text-list-100-sec.npy", type=str,
                        help='Path to corpus')
    parser.add_argument('--sparse_passage_dir', default="indexes/enwiki-dec2021/", type=str,
                        help='Path to sparse passage index')
    parser.add_argument('--dense_passage_dir', default=None, type=str,
                        help='Path to dense passage index')
    parser.add_argument('--lm_model', default="Trelis/Llama-2-7b-chat-hf-hosted-inference-8bit", type=str,
                        help='LLM')
    parser.add_argument('--retrieval_qry_model', default="facebook/dragon-plus-query-encoder", type=str,
                        help='Retrieval query encoder')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--threads', default=32, type=int,
                        help='Number of threads for CPU retrieval')
    parser.add_argument('--nprobe', default=4096, type=int,
                        help='Number of probes for Faiss retrieval')
    parser.add_argument('--num_passages', default=40, type=int,
                        help='Number of passages to retrieve')
    parser.add_argument('--num_tokens', default=1024, type=int,
                        help='Number of tokens to retrieve')
    parser.add_argument('--pad_id', default=-1, type=int,
                        help='Padding token id')
    parser.add_argument('--window', default=64, type=int,
                        help='Number of tokens to use for draft proposal')
    parser.add_argument('--alpha', default=0.3, type=float,
                        help='Alpha (RRC)')
    parser.add_argument('--tau', default=0.1, type=float,
                        help='TAU (RRC)')
    parser.add_argument('--gamma', default=5e-4, type=float,
                        help='Gamma (Relaxed Spec Decoding)')
    parser.add_argument('--beta', default=0.5, type=float,
                        help='Beta (Dynamic span selection)')
    parser.add_argument('--fusion_coef', default=0.3, type=float,
                        help='Fusion coef for first-stage retrieval')
    parser.add_argument('--pre_tokenized', action='store_true')
    parser.add_argument('--pre_retrieved', action='store_true')
    parser.add_argument('--start_tag', default="", type=str,
                        help='Start tag of prompt. Default Llama-2-chat.')
    parser.add_argument('--end_tag', default="", type=str,
                        help='Start tag of prompt. Default Llama-2-chat.')
    parser.add_argument('--prefix_len', default=128, type=int,
                        help='Prefix length for ppl eval.')
    parser.add_argument('--target_len', default=256, type=int,
                        help='Target length for ppl eval.')
    parser.add_argument('--n_batches', default=50, type=int,
                        help='Number of batches for ppl eval.')
    args = parser.parse_args()
    task_evaluation(args)
