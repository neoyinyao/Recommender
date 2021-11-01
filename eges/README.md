An implementation of **EGES**(Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba) with **DGL** library working with **Tensorflow** backend 

paper: https://arxiv.org/pdf/1803.02349.pdf
dataset: http://jmcauley.ucsd.edu/data/amazon/  Amazon Electronics 

Run code: python train.py --model_type EGES

evaluation result(link prediction auc):
Base Graph Embedding(DeepWalk) : **0.8901** 

Enhanced Graph Embedding : **0.9547**

Enhanced Graph Embedding with Side information : **0.9576**

**Cold-Start** problem:

deal out-of-vocabulary item, by average default_item_id_embedding and default_item_side_embedding