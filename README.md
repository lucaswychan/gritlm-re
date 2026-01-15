<div align="center">

# Do Reasoning Models Enhance Embedding Models? 
## (Training Codes Repository)

<p align="center">🏆  <a href="https://github.com/lucaswychan">Arxiv Paper</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/collections/lucaswychan/reasoning-embedding">Hugging Face</a></p> 


<img src="assests/re-thumbnail.png" alt="thumbnail" width="400">

</div>

This repository is the training code for the embedding model used in the paper 'Do Reasoning Models Enhance Embedding Models'. We greatly thanks the work from [Generative Representational Instruction Tuning](https://arxiv.org/abs/2402.09906). In the original GritLM implementation, they consider both embedding and generative training setting. In our work, however, we discard all generative settings as it is not necessary to train the embedding models.

## Installation

We use `uv` to manage the dependencies. FlashAttention should be separately built as using `uv sync` to build is troublesome.  
[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install flash-attn --no-build-isolation
```

If you want to use GradCache, you need to use the one in this repository
```bash 
cd gritlm/training/GradCache
uv pip install -e .
cd ../..
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
