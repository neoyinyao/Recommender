from tensorflow import keras
import dgl
import dgl.function as fn

from layers import FeatureProjector, SageNet


class PinSageModel(keras.Model):
    def __init__(self, full_graph, itype, num_layers, embedding_size, conv_hidden_size, conv_output_size):
        super().__init__()
        self.feature_projector = FeatureProjector(full_graph, itype, embedding_size)
        self.sagenet = SageNet(num_layers, conv_hidden_size, conv_output_size)

    def item2item_scorer(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))  # use dot product as score
            scores = graph.edata['score']
        return scores

    def call(self, pos_graph, neg_graph, blocks):
        """
        pos_graph: subgraph of train_g,including pos_edges from heads to pos_tails and all train_g nodes
        neg_graph: subgraph of train_g,including neg_edges from heads to neg_tails and all train_g nodes
        blocks: subgraph of minibatch nodes and all its random walk sampled neighborhoods
        """
        hidden_repr = self.get_repr(blocks)
        pos_score = self.item2item_scorer(pos_graph, hidden_repr)  # None
        neg_score = self.item2item_scorer(neg_graph, hidden_repr)  # None
        return pos_score, neg_score

    def get_repr(self, blocks):
        src_induced_ids = blocks[0].srcdata[dgl.NID]  # 整图的节点索引
        # dst_induced_ids = blocks[-1].dstdata[dgl.NID]
        hidden_src = self.feature_projector(src_induced_ids)  # 初始化的feature
        # hidden_dst = self.feature_projector(dst_induced_ids)
        hidden_repr = self.sagenet(blocks, hidden_src)  # sagenet representation
        # hidden_repr = tf.concat([hidden_dst, hidden_repr], axis=-1)
        return hidden_repr
