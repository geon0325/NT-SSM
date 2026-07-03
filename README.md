# Rethinking Contrastive Learning in Graph Collaborative Filtering: Limitations and A Simple Remedy

[![arXiv](https://img.shields.io/badge/arXiv-2605.24015-b31b1b.svg)](https://arxiv.org/abs/2605.24015)

Official implementation of **NT-SSM**, a neighbor-type–aware contrastive learning objective for graph collaborative filtering (GCF), from our ICML 2026 paper:

> **Rethinking Contrastive Learning in Graph Collaborative Filtering: Limitations and a Simple Remedy**
> Geon Lee, Sunwoo Kim, Kyungho Kim, Kijung Shin (KAIST)
> [arXiv:2605.24015](https://arxiv.org/abs/2605.24015)

## Overview

GCF models such as LightGCN compute a user–item score by implicitly aggregating similarities over a huge number of multi-hop *neighbor pairs*, so how contrastive learning (CL) chooses which neighbor pairs to upweight during training turns out to matter a lot for recommendation quality. We show that the widely-used Sampled Softmax (SSM) loss upweights neighbor pairs based only on item-side structural similarity, ignoring the user side, and applies the same rule to every type of neighbor pair (user–user, item–item, user–item, item–user) even though the best strategy differs across types.

**NT-SSM** fixes both issues with a bidirectional objective and per-type coefficients that adaptively control how strongly each neighbor-pair type gets upweighted. Across four datasets (LastFM, MovieLens-1M, Yelp2018, Amazon-Book) and three GCF backbones (LightGCN, SimGCL, NCL), NT-SSM consistently improves over SSM, with double-digit NDCG@20 gains on several dataset/backbone combinations, at a modest training-time overhead and no extra learnable parameters. Full results and derivations are in the paper.

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

You can replace `LightGCN_NT` with `SimGCL_NT`, `NCL_NT`, or `NGCF_NT` to train the other backbones with NT-SSM. Dropping the `_NT` suffix (e.g. `LightGCN`, `SimGCL`, `NCL`, `NGCF`) trains the corresponding backbone with the standard SSM loss instead.

The `alpha_uu`, `alpha_ii`, `alpha_ui`, and `alpha_iu` flags are the neighbor-type-specific coefficients (α) described in the paper, controlling how aggressively user–user, item–item, user–item, and item–user neighbor pairs are upweighted, respectively.

See [`run.sh`](run.sh) and [`run2.sh`](run2.sh) for example sweeps over these coefficients and the temperature (`tau`).

Training logs are written to `logs/` and learned embeddings to `embs/`, both created automatically on first run.

## Repository Structure

```
.
├── main.py            # Entry point (CLI arguments, dataset/GPU setup)
├── SELFRec.py          # Loads data and dispatches to the selected model
├── base/                # Base recommender / graph-recommender / torch interface classes
├── data/                # Data loading and graph construction utilities
├── model/graph/         # Model implementations (LightGCN, NGCF, SimGCL, NCL, and their _NT variants)
├── util/                # Losses, samplers, evaluation, and misc. utilities
├── dataset/              # Train/valid/test splits for LastFM, ML-1M, Yelp2018, Amazon-Book
├── run.sh, run2.sh      # Example hyperparameter sweep scripts
└── requirements.txt
```

This codebase builds on the [SELFRec](https://github.com/Coder-Yu/SELFRec) framework.

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