import numpy as np
import dgl
import tensorflow as tf


def item2item_batch_sampler(g, utype, itype, batch_size):
    item_candidates = g.number_of_nodes(ntype=itype)
    while True:
        heads = np.random.randint(0, item_candidates, size=(batch_size,))
        item2user_etype = list(g.metagraph()[itype][utype])[0]
        user2item_etype = list(g.metagraph()[utype][itype])[0]
        metapath = [item2user_etype, user2item_etype]
        pos_tails = dgl.sampling.random_walk(g, nodes=heads, metapath=metapath)[0][:, 2]
        neg_tails = np.random.randint(0, item_candidates, size=(batch_size,))
        mask = (pos_tails != -1)
        heads = tf.constant(heads)
        neg_tails = tf.constant(neg_tails)
        yield heads[mask], pos_tails[mask], neg_tails[mask]


class PinSageSampler(object):
    def __init__(self, g, itype, utype, num_layers, random_walk_length, num_random_walks, termination_prob,
                 num_neighbors, weight_column='weight'):
        self.g = g
        self.num_layers = num_layers
        self.sampler = dgl.sampling.PinSAGESampler(g, itype, utype, random_walk_length, termination_prob,
                                                   num_random_walks, num_neighbors, weight_column)

    def generate_blocks(self, seeds, heads=None, pos_tails=None, neg_tails=None):
        blocks = []
        dst_nodes = seeds
        for i in range(self.num_layers):
            frontier = self.sampler(dst_nodes)
            if heads is not None:
                eids = frontier.edge_ids(tf.concat([heads, heads], axis=0), tf.concat([pos_tails, neg_tails], axis=0),
                                         return_uv=True)[2]
                #             print(eids)
                if len(eids) > 0:  # remove heads,pos_tail edge and heads,neg_tail edge,避免信息泄露
                    frontier.remove_edges(eids)
            block = dgl.to_block(frontier, dst_nodes)
            dst_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks

    def sample_from_item_pairs(self, heads, pos_tails, neg_tails, itype):
        pos_graph = dgl.graph((heads, pos_tails), num_nodes=self.g.number_of_nodes(itype))
        neg_graph = dgl.graph((heads, neg_tails), num_nodes=self.g.number_of_nodes(itype))
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])  # fileter isolated nodes
        seeds = pos_graph.ndata[dgl.NID]  # 整图的节点id
        blocks = self.generate_blocks(seeds, heads, pos_tails, neg_tails)
        return pos_graph, neg_graph, blocks
