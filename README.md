# Rethinking Contrastive Learning in Graph Collaborative Filtering: Limitations and A Simple Remedy

This repository contains the official implementation for **NT-SSM**, an effective and principled contrastive learning objective for graph collaborative filtering. 


## Datasets
Datasets are available in the [dataset] folder.


## Training GCF Models

To train LightGCN with NT-SSM, exceute:
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

You can replace LightGCN_NT with SimGCL_NT and NCL_NT.
