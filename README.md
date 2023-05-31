# GraphDA
This is our Pytorch implementation for the paper:
You may also find it on [Arxiv](https://arxiv.org/pdf/2304.03344.pdf)

Please cite our paper if you use the code:
```bibtex
@inproceedings{fan2023graphda,
author = {Fan, Ziwei and Xu, Ke and Dong, Zhang  and Peng, Hao and Zhang, Jiawei and Yu, Philip S.},
title = {Graph Collaborative Signals Denoising and Augmentation for Recommendation},
year = {2023},
booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
series = {SIGIR 2023}
}
```

## Paper Abstract
Graph collaborative filtering (GCF) is a popular technique for capturing high-order collaborative signals in recommendation systems. However, GCF's bipartite adjacency matrix, which defines the neighbors being aggregated based on user-item interactions, can be noisy for users/items with abundant interactions and insufficient for users/items with scarce interactions. Additionally, the adjacency matrix ignores user-user and item-item correlations, which can limit the scope of beneficial neighbors being aggregated. 

In this work, we propose a new graph adjacency matrix that incorporates user-user and item-item correlations, as well as a properly designed user-item interaction matrix that balances the number of interactions across all users. To achieve this, we pre-train a graph-based recommendation method to obtain users/items embeddings, and then enhance the user-item interaction matrix via top-K sampling. We also augment the symmetric user-user and item-item correlation components to the adjacency matrix. Our experiments demonstrate that the enhanced user-item interaction matrix with improved neighbors and lower density leads to significant benefits in graph-based recommendation. Moreover, we show that the inclusion of user-user and item-item correlations can improve recommendations for users with both abundant and insufficient interactions. The code is in https://github.com/zfan20/GraphDA.

## Code introduction and Environment Setup
The code is implemented based on [LightGCN-Pytorch](https://github.com/gusye1234/LightGCN-PyTorch). Other codes are incorporated correspondingly. Follow the LightGCN-Pytorch to create the environment.

## Datasets
We use the Amazon Review datasets. The data split is done in the leave-one-out setting for all users. Make sure you download the datasets from the [link](https://jmcauley.ucsd.edu/data/amazon/).

### Data Preprocessing
Please refer to other works in sequential recommendation preprocessing for the leave-one-out setting.

## Pre-training and Prediction
```
python pretrain.py --data_name=Toys_and_Games --dropout=1 --lr=0.001 --recdim=128 --layer=4 --decay=0.0001 --keepprob=0.7 --model_name=LightGCN
```

## Enhanced-UI
```
python distill_separate.py --data_name=Toys_and_Games --dropout=1 --lr=0.001 --recdim=128 --distill_layers=3 --decay=0.001 --keepprob=0.3 --model_name=LightGCN --distill_userK=5 --distill_itemK=7 --distill_thres=0.5
```

## GraphDA
```
python distill_separate_uuii.py --data_name=Toys_and_Games --dropout=1 --lr=0.001 --recdim=128 --distill_layers=1 --decay=0.001 --keepprob=0.3 --model_name=LightGCN --distill_userK=5 --distill_itemK=9 --distill_uuK=3 --distill_iiK=5 --distill_thres=0.5 --uuii_thres=-1
```

## Grid Search
I did the grid search to obtain the best validation mrr for all models either in the pre-training, Enhanced-UI and GraphDA. Make sure you put your best validation mrr model in checkpoints folder before doing the distillation. For the distillation, several hyper-parameters are also searched. 
