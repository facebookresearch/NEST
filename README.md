# Nearest Neighbor Speculative Decoding for LLM Generation and Attribution
This is the official implementation of [Nearest Neighbor Speculative Decoding for LLM Generation and Attribution, NeurIPS 2024](https://arxiv.org/abs/2405.19325) using Huggingface.
<img width="1108" alt="Screenshot 2025-04-22 at 10 35 41 PM" src="https://github.com/user-attachments/assets/90819320-6a78-49e3-b03f-39dead39bef8" />

## Dependencies
First, make sure you have [Anaconda3](https://docs.anaconda.com/anaconda/install/index.html) installed.
Then use conda to create a new environment and activate it:
```
conda create -n nest python=3.10
conda activate nest
```
Now let's install the packages. First, follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch on your machine.
Then install faiss:
```
conda install faiss
```
Finally install the packages in `requirements.txt`. Remember to comment out the packages in the .txt file that you've already installed to avoid conflicts.
```
pip install -r requirements.txt
```
For flash attention, run
```
pip install flash-attn --no-build-isolation
```

## Demo
To get started, we provide an example script of using NEST for generation:
```
python demo.py
```
You can play with the hyperparameters in the demo to understand how they affect the NEST generation.

## Models
The default models we use are Llama-2-chat-7b, 13b, and 70b. You can switch to other CausalLM models on Huggingface. Remember to change the `start_tag` and `end_tag` value for your models. We provide the links of the Llama-2-chat models we used in the following:
| Models    | Huggingface Tag |
| -------- | ------- |
| Llama-2-chat-7b  | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)    |
| Llama-2-chat-13b | [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)     |
| Llama-2-chat-70b    | [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)    |

Remeber to get the authorization from Meta before using the above models.

## Data Prep
### Corpus
We use the Wikipedia 2021 dump from the [Atlas repo](https://github.com/facebookresearch/atlas?tab=readme-ov-file#available-data-and-models-for-download). Download the corpus following the instructions in the repo.
### Indexes
The default embedders/retrievers we used are [DRAGON+](https://github.com/facebookresearch/dpr-scale?tab=readme-ov-file) and BM25 ([Pyserini](https://github.com/castorini/pyserini)).
To build the sparse index using BM25, run:
```
python data/convert_atlas_corpus_to_pyserini_format.py your/path/to/downloaded/corpus collections/enwiki-dec2021
bash data/pyserini_index.sh
``` 
For the dense index, we use FAISS and the index string "IVF65536,PQ256" to build the index. Please see the DRAGON+ repo for more detailed index building instructions.
We also open-source the dense index we used.
| Indexes    | Size |
| -------- | ------- |
| [DRAGON+](https://dl.fbaipublicfiles.com/wiki2021_dragon_index/passage.ivf65536.pq256.index) | 8.96GB     |
| BM25 (Pyserini)  |  3.48GB   |

By default, both BM25 and DRAGON+ run on CPUs during retrieval of which the latency is controlled by the number of threads. However, for the dense index, it is also possible to use GPUs to accelerate which we leave for other custom implementations.

### Tasks
| Task    | Description | Size |
| -------- | ------- | ------- |
| [WikiText-103](https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/wikitext-103.tar.gz)  |  Text completion  | 2357 (test) |
| [NQ](https://vault.cs.uwaterloo.ca/s/C4AreqGKP5YaXmG)  |  Question Answering  | 3610 (test) |
| [TriviaQA](https://vault.cs.uwaterloo.ca/s/5GEkEWYkAqFmMsq)  |  Question Answering  | 11313 (test) |
| [HotpotQA](https://www.kaggle.com/code/jeromeblanchet/hotpotqa-data-analysis/data?select=hotpot_dev_fullwiki_v1.json)  |  Question Answering  | 2500 (dev, sub-sample) |
| [MedMCQA](https://github.com/MedMCQA/MedMCQA)  |  Question Answering  | 2500 (dev, sub-sample) |
| [TruthfulQA](https://github.com/sylinrl/TruthfulQA)  |  Question Answering  | 817 (test) |
| [FactScore](https://github.com/shmsw25/FActScore)  |  Fact Verification  | 500 (unlabeled) |

We evaluate NEST on the above tasks based on the Wikipedia corpus.
To preprocess the data, we provide a script in the `data/` folder for each task.

To evaluate the NEST on the above tasks, run
```
bash scripts/eval_gen_task.sh
```
Change the `task` argument separated by "," and other arguments before running the evaluation. Note the Llama-2-chat models use special tags ("\[INST\] and \[/INST\]") for instructions fine-tuning. These tags might not be necessary if you use other CausalLMs.

Due to the compatibility issues between the CC-BY-NC license of NEST's code and the GPL3 license of [MAUVE](https://github.com/krishnap25/mauve), we do not include this metric in the evaluation on WikiText-103 and Pile-of-Law.

For [FactScore](https://github.com/shmsw25/FActScore), please take the generation results and follow the instruction in the FactScore repo for evaluation. In the paper, we use the internal LLM finetuned for fact decomposition and fact checking, which may not be released in the future. 

### Pre-retrieved
You can also pre-fectch the supporting documents (as we provided in the task data) to avoid passage retrieval during generation by running:
```
bash scripts/eval_retrieval_task.sh
```
The results will be saved in the same format as the input data with an extra field "support". Remeber to move the results to the data input path before running evaluation. You need to add the `--pre_retrieved` argument for document pre-fetching.

### Pre-tokenized
You can also pre-encode the corpus into tokens to save time during generation. You need to add the `--pre_tokenized` argument for corpus pre-encoding. See the `encode` function in `knn_transformer.py` for more details of pre-encoding.

## Citation
```
@misc{li2024nearestneighborspeculativedecoding,
      title={Nearest Neighbor Speculative Decoding for LLM Generation and Attribution}, 
      author={Minghan Li and Xilun Chen and Ari Holtzman and Beidi Chen and Jimmy Lin and Wen-tau Yih and Xi Victoria Lin},
      year={2024},
      eprint={2405.19325},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.19325}, 
}
```

## License
The code of NEST is licensed under CC-BY-NC.
