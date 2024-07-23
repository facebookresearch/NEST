# Nearest Neighbor Speculative Decoding for LLM Generation and Attribution
This page describes how to implement [NEST](https://arxiv.org/abs/2405.19325) with dpr-scale.
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

## Data Prep
### Corpus
Wikipedia 2021 dump
### Indexes
Dense and Sparse
### Tasks
WikiText-103, NQ, TriviaQA, HotpotQA, MedMCQA, Truthfulqa, FactScore


## Generation
```
python scripts/test_generation.py
```
## Evaluation
```
bash scripts/eval.sh
```
## License
The code of NEST is licensed under CC-BY-NC.