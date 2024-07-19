from tasks import WikiText103Task, FactScoreTask, NQTask, TriviaQATask, MedMCQATask, HotPotQATask, TruthfulQATask
from knn_transformers import KNNTransformer
from generation import generate
from utils import load_datasets

from collections import defaultdict
import os
import argparse
import json
import jsonlines

TASK = {
    "wikitext-103":WikiText103Task,
    "factscore":FactScoreTask,
    "nq":NQTask,
    "triviaqa":TriviaQATask,
    "medmcqa":MedMCQATask,
    "hotpotqa":HotPotQATask,
    "truthfulqa":TruthfulQATask,
}

def eval_generation(args, model, dataset, task_str):
    task = TASK[task_str](start_tag=args.start_tag, end_tag=args.end_tag)
    results = defaultdict(list)
    outputs = []
    for i, data in enumerate(dataset):
        print(f"Batch {i}")
        queries = task.get_query(data)
        query_inputs = model.passage_tokenizer(queries, return_tensors='pt', padding=True)
        _, batch_doc_ids = model.retrieve_passage(queries, query_inputs)
        model.build_token_index(batch_doc_ids)

        prompt = task.get_prompt(data)
        tokens, _, _ = generate(model,
                                prompt,
                                max_prompt_len=args.prompt_len,
                                max_gen_len=args.max_len,
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
            retrieval_ctx_model=args.retrieval_ctx_model,
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

    if args.tasks is None:
        print("No tasks to eval!")
        return
    else:
        tasks = args.tasks.split(",")
    
    for task_str in tasks:
        output_dir = "/".join([args.output_dir, task_str])
        os.makedirs(output_dir, exist_ok=True)
        print(f"Evaluating {task_str}...")
        dataset = load_datasets("/".join([args.data_dir, task_str, f"{args.mode}.10.jsonl"]), batch_size=args.batch_size)
        metrics, outputs = eval_generation(args, model, dataset, task_str)
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
    parser.add_argument('--retrieval_ctx_model', default="facebook/dragon-plus-context-encoder", type=str,
                        help='Retrieval context encoder')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--threads', default=32, type=int,
                        help='Number of threads for CPU retrieval')
    parser.add_argument('--nprobe', default=4096, type=int,
                        help='Number of probes for Faiss retrieval')
    parser.add_argument('--num_passages', default=10, type=int,
                        help='Number of passages to retrieve')
    parser.add_argument('--num_tokens', default=512, type=int,
                        help='Number of tokens to retrieve')
    parser.add_argument('--pad_id', default=-1, type=int,
                        help='Padding token id')
    parser.add_argument('--window', default=64, type=int,
                        help='Number of tokens to use for draft proposal')
    parser.add_argument('--prompt_len', default=256, type=int,
                    help='Max context length')
    parser.add_argument('--max_len', default=256, type=int,
                        help='Max generation length')
    parser.add_argument('--alpha', default=0.2, type=float,
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
    parser.add_argument('--start_tag', default="[INST]", type=str,
                        help='Start tag of prompt. Default Llama-2-chat.')
    parser.add_argument('--end_tag', default="[/INST]", type=str,
                        help='Start tag of prompt. Default Llama-2-chat.')
    args = parser.parse_args()
    task_evaluation(args)
