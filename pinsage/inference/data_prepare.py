import os
import json
import pickle
import sys

sys.path.append('../train')
os.environ['DGLBACKEND'] = 'tensorflow'

import tensorflow as tf
import dgl
import numpy as np

from train import PinSageModel


def build_util():
    with open('../data/total_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    total_g = dataset['total-graph']
    utype = dataset['user-type']
    itype = dataset['item-type']

    num_layers = 2
    convolve_hidden_size, convolve_output_size = 32, 16
    train_batch_size = 4
    random_walk_length, num_random_walks, termination_prob, num_neighbors = 2, 4, 0, 3
    embedding_size = 8
    sampler = dgl.sampling.PinSAGESampler(total_g, itype, utype, random_walk_length, termination_prob, num_random_walks,
                                          num_neighbors)
    dataset = tf.data.Dataset.range(total_g.num_nodes(itype)).batch(train_batch_size)
    with open('../data/item-neighbors.json', 'w') as f:
        item_neighbors = []
        for item_batch in dataset:
            frontier = sampler(item_batch)
            neighbor_nodes, src_nodes = frontier.edges()
            weights = frontier.edata['weights']
            for src, neighbor, weight in zip(src_nodes, neighbor_nodes, weights):
                example = {}
                example['item_id'] = int(src.numpy().item())
                example['neighbor'] = int(neighbor.numpy().item())
                example['weight'] = int(weight.numpy().item())
                item_neighbors.append(example)
        json.dump(item_neighbors, f)
    total_g.nodes[itype].data['id'] = tf.constant(np.arange(total_g.number_of_nodes(itype)))
    model = PinSageModel(total_g, itype, num_layers, embedding_size, convolve_hidden_size, convolve_output_size)
    with open('../data/item-features.json', 'w') as f:
        item_features = []
        for item_batch in dataset:
            batch_features = model.feature_projector(item_batch)  # None,embedding_size
            for item_id, feature in zip(item_batch, batch_features):
                example = {
                    'item_id': int(item_id.numpy().item()),
                    'feature': feature.numpy().tolist()
                }
                item_features.append(example)
        json.dump(item_features, f)


if __name__ == '__main__':
    build_util()
