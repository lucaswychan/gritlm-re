<div align="center">

# Do Reasoning Models Enhance Embedding Models? 
## (Training Codes Repository)

<p align="center">🏆  <a href="https://arxiv.org/abs/2601.21192">Arxiv Paper</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/collections/lucaswychan/reasoning-embedding">Hugging Face</a></p> 


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
# Method 1: For transformers<5.0.0 (default in requirements.txt.)
cp models/modeling_qwen2_v4.py .venv/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py
cp models/modeling_qwen3_v4.py .venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py

# Method 2: For transformers>=5.0.0 (if you somehow modified the training codes that is adapt to transformers>=5.0.0)
cp models/modeling_qwen2.py .venv/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py
cp models/modeling_qwen3.py .venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py
```

## Embedding Model Training

To train the model, simply run

```bash
# under train/gritlm-re
bash scripts/training/train_embonly.sh
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
