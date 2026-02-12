<div align="center">

# Do Reasoning Models Enhance Embedding Models? 
## (Training Codes Repository)

<p align="center">
  <a href="https://arxiv.org/abs/2601.21192">
    <img alt="ArXiv" src="https://img.shields.io/badge/Paper-ArXiv%3A2601.21192-b31b1b.svg?style=flat-rounded&logo=arxiv&logoColor=white">
  </a>
  <a href="https://huggingface.co/collections/lucaswychan/reasoning-embedding">
    <img alt="Hugging Face Collection" src="https://img.shields.io/badge/HF-Reasoning--Embedding-blueviolet?style=flat-rounded&logo=huggingface">
  </a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue.svg?style=flat-rounded&logo=python">
  </a>
</p>


<img src="assests/re-thumbnail.png" alt="thumbnail" width="400">

</div>

This repository is the training code for the embedding model used in the paper 'Do Reasoning Models Enhance Embedding Models'. We greatly thanks the work from [Generative Representational Instruction Tuning](https://arxiv.org/abs/2402.09906). In the original GritLM implementation, they consider both embedding and generative training setting. In our work, however, we discard all generative settings as it is not necessary to train the embedding models.

## Abstract

State-of-the-art embedding models are increasingly derived from decoder-only Large Language Model (LLM) backbones adapted via contrastive learning. Given the emergence of reasoning models trained via Reinforcement Learning with Verifiable Rewards (RLVR), a natural question arises: do enhanced reasoning translate to superior semantic representations when these models serve as embedding initializations? Contrary to expectation, our evaluation on MTEB and BRIGHT reveals a **null effect**: embedding models initialized from RLVR-tuned backbones yield no consistent performance advantage over their base counterparts when subjected to identical training recipes. To unpack this paradox, we introduce **H**ierarchical **R**epresentation **S**imilarity **A**nalysis (HRSA), a framework that decomposes similarity across representation, geometry, and function levels. HRSA reveals that while RLVR induces irreversible latent manifold's local geometry reorganization and reversible coordinate basis drift, it preserves the global manifold geometry and linear readout. Consequently, subsequent contrastive learning drives strong alignment between base- and reasoning-initialized models, a phenomenon we term **Manifold Realignment**. Empirically, our findings suggest that unlike Supervised Fine-Tuning (SFT), RLVR optimizes trajectories within an existing semantic landscape rather than fundamentally restructuring the landscape itself.

## Installation

Clone the repository and initialize the submodule:

```bash
git clone https://github.com/lucaswychan/gritlm-re.git
cd gritlm-re
```

Install dependencies using either `uv` (recommended) or `pip`:

```bash
# Method 1: Using uv (Install uv first: https://docs.astral.sh/uv/getting-started/installation/)
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install flash-attn --no-build-isolation

# Method 2: Using pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

If you want to use GradCache, you need to use the one in this repository
```bash 
cd gritlm/training/GradCache
uv pip install -e .
cd ../..
```

**⚠️ Important:** Enable bidirectional attention (required for embedding models):

```bash
# Method 1: For transformers<5.0.0 (default in requirements.txt)
cp models/modeling_qwen2_v4.py .venv/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py
cp models/modeling_qwen3_v4.py .venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py

# Method 2: For transformers>=5.0.0 (if you somehow modified the training codes that is adapt to transformers>=5.0.0)
cp models/modeling_qwen2.py .venv/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py
cp models/modeling_qwen3.py .venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py
```

## Embedding Model Training

To train the model, simply run

```bash
bash scripts/training/train_embonly.sh
```

Change [`output_dir`](scripts/training/train_embonly.sh#L62) and [`model_name_or_path`](scripts/training/train_embonly.sh#L63) to the corresponding output directory and base model name, respectively.

## Dataset

### Download

We provided the pretraining dataset used in our paper, see [here](https://huggingface.co/collections/lucaswychan/reasoning-embedding).

You should first download all the datasets into the same directory, and then specify the path to that directory as `TRAIN_DATA` in [`scripts/training/train_embonly.sh`](scripts/training/train_embonly.sh#L58). You can also build your own dataset by following the below format instruction.

### Format

The dataset is structured according to the [`GritLM`](https://github.com/lucaswychan/gritlm-re) repository's format: `{"query": List[str], "pos": List[str], "neg": List[str]}`. The script to mine the hard negatives is [here](https://github.com/HKUST-KnowComp/Reasoning-Embedding/blob/main/datasets/mine_hard_neg.py).

*   **`query`**: This is a list containing two strings.
    *   `query[0]` holds the instruction. A complete list of instructions can be found [here](https://github.com/HKUST-KnowComp/Reasoning-Embedding/blob/main/evaluation/task_prompts.json).
    *   `query[1]` contains the actual query text.
*   **`pos`**: A list with a single string, representing the positive anchor for the query. You can add more anchors to the list.
*   **`neg`**: A list containing 1 - 3 strings, which are the mined hard negatives associated with the query.

For example,

```json
{
    "query": [
        "Instruct: Given a premise, retrieve a hypothesis that is entailed by the premise\nQuery: ",
        "A woman wearing a green and pink dress is dancing with someone wearing a blue top with white pants."
    ],
    "pos": [
        "The woman in green and pink is dancing."
    ],
    "neg": [
        "The dancing woman is alone in her bedroom.",
        "A woman in a dress dances with a man.",
        "A woman wearing a green belly dancing outfit is near a man and woman who are seated"
    ]
}
```

### Citation

Please cite our paper 😊
```bibtex
@misc{chan2026reasoningmodelsenhanceembedding,
      title={Do Reasoning Models Enhance Embedding Models?}, 
      author={Wun Yu Chan and Shaojin Chen and Huihao Jing and Kwun Hang Lau and Elton Chun-Chai Li and Zihao Wang and Haoran Li and Yangqiu Song},
      year={2026},
      eprint={2601.21192},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.21192}, 
}
```

And also cite the original paper of GritLM 😊
```bibtex
@misc{muennighoff2024generative,
      title={Generative Representational Instruction Tuning}, 
      author={Niklas Muennighoff and Hongjin Su and Liang Wang and Nan Yang and Furu Wei and Tao Yu and Amanpreet Singh and Douwe Kiela},
      year={2024},
      eprint={2402.09906},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
