# Rethinking Contrastive Learning in Graph Collaborative Filtering: Limitations and A Simple Remedy

[![arXiv](https://img.shields.io/badge/arXiv-2605.24015-b31b1b.svg)](https://arxiv.org/abs/2605.24015)

Official implementation of **NT-SSM**, a neighbor-type–aware contrastive learning objective for graph collaborative filtering (GCF), from our [ICML 2026](https://icml.cc/Conferences/2026) paper:

> **Rethinking Contrastive Learning in Graph Collaborative Filtering: Limitations and a Simple Remedy**
> Geon Lee, Sunwoo Kim, Kyungho Kim, Kijung Shin (KAIST)
> [arXiv:2605.24015](https://arxiv.org/abs/2605.24015)

## Overview

Graph collaborative filtering (GCF) predicts user–item relevance by aggregating interactions over a large number of multi-hop neighbor pairs. We show that the widely used Sampled Softmax (SSM) loss induces suboptimal neighbor-pair update dynamics by relying primarily on item-side structural similarity and treating all neighbor-pair types identically. We propose **NT-SSM**, a neighbor-type-aware contrastive objective that jointly considers user- and item-side structural similarity and adaptively controls updates for different neighbor-pair types. NT-SSM consistently improves recommendation performance across multiple datasets and GCF backbones with only modest training overhead and no additional learnable parameters.

## Requirements

```bash
pip install -r requirements.txt
```

## Datasets

Datasets are available in the [`dataset`](dataset) folder: `lastfm`, `ml-1m`, `yelp2018`, and `amazon-book`. Each dataset directory contains `train.txt`, `valid.txt`, and `test.txt`.

## Training GCF Models

To train LightGCN with NT-SSM, execute:
```bash
python main.py \
    --gpu 0 \
    --dataset ml-1m \
    --model_name LightGCN_NT \
    --model_type graph \
    --item_ranking 10,20,40 \
    --embedding_size 64 \
    --epoch 200 \
    --batch_size 2048 \
    --learning_rate 0.001 \
    --reg_lambda 0.0001 \
    --n_layer 2 \
    --alpha_uu 1.0 \
    --alpha_ii 1.0 \
    --alpha_ui 1.0 \
    --alpha_iu 1.0 \
    --loss_type ssm \
    --tau 0.2
```

You can replace `LightGCN_NT` with `SimGCL_NT` or `NCL_NT` to train the corresponding backbone with **NT-SSM**. Removing the `_NT` suffix (e.g., `LightGCN`, `SimGCL`, or `NCL`) trains the backbone with the standard **SSM** loss.

The `alpha_uu`, `alpha_ii`, `alpha_ui`, and `alpha_iu` arguments correspond to the four neighbor-pair types—user–user (UU), item–item (II), user–item (UI), and item–user (IU)—and control their type-specific update dynamics.

See [`run.sh`](run.sh) for an example training script on **MovieLens-1M** using **LightGCN + NT-SSM**.

Training logs and learned embeddings are automatically saved to the `logs/` and `embs/` directories, respectively.

## Repository Structure

```text
.
├── main.py             # Entry point (CLI arguments, dataset/GPU setup)
├── SELFRec.py          # Loads data and dispatches to the selected model
├── base/               # Base recommender, graph recommender, and PyTorch interface classes
├── data/               # Data loading and graph construction utilities
├── model/graph/        # Model implementations (LightGCN, SimGCL, NCL, and their _NT variants)
├── util/               # Losses, samplers, evaluation, and miscellaneous utilities
├── dataset/            # Train/validation/test splits for LastFM, ML-1M, Yelp2018, and Amazon-Book
├── run.sh              # Example script
└── requirements.txt    # Python dependencies
```


## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{lee2026rethinking,
  title     = {Rethinking Contrastive Learning in Graph Collaborative Filtering: Limitations and a Simple Remedy},
  author    = {Lee, Geon and Kim, Sunwoo and Kim, Kyungho and Shin, Kijung},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## Acknowledgement

This code is implemented based on [https://github.com/Coder-Yu/SELFRec](SELFRec).
