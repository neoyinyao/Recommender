import numpy as np
import tensorflow as tf
from tensorflow import keras
import dgl.function as fn


class Convolve(keras.layers.Layer):
    def __init__(self, conv_hidden_size, conv_output_size):
        super().__init__()
        self.fc_1 = keras.layers.Dense(conv_hidden_size, use_bias=True, activation=tf.nn.relu)
        self.fc_2 = keras.layers.Dense(conv_output_size, use_bias=True, activation=tf.nn.relu)

    def call(self, block, h):
        h_src, h_dst = h
        h_src = self.fc_1(h_src)  # neighbor transformation
        block.srcdata['u'] = h_src
        block.edata['w'] = tf.cast(block.edata['weight'], h_src.dtype)
        block.update_all(fn.u_mul_e('u', 'w', 'v'), fn.sum('v', 'vs'))  # weighted aggregate neighborhoods info
        block.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))  # sum neighborhoods weights
        vs = block.dstdata['vs']  # None,conv_hidden_size
        ws = block.dstdata['ws']  # None
        # clip weight,because 'ws' could be zero,divide zero raise tf inf gradients
        ws = tf.clip_by_value(ws, 1, np.inf)
        ws = tf.expand_dims(ws, axis=-1)
        nv = vs / ws  # None,conv_hidden_size, importance pooling/aggregating
        new = tf.concat([nv, h_dst], axis=-1)
        new = self.fc_2(new)  # concat transformation
        l2norm = tf.norm(new)
        new = new / l2norm  # l2 normalization
        return new


class SageNet(keras.layers.Layer):
    def __init__(self, num_layers, conv_hidden_size, conv_output_size):
        super().__init__()
        self.convolves = [Convolve(conv_hidden_size, conv_output_size) for _ in range(num_layers)]
        self.fc_1 = keras.layers.Dense(conv_hidden_size, activation=tf.nn.relu, use_bias=True)
        self.fc_2 = keras.layers.Dense(conv_output_size)

    def call(self, blocks, h_src):
        for convolve, block in zip(self.convolves, blocks):
            h_dst = h_src[:block.num_dst_nodes()]
            h_src = convolve(block, (h_src, h_dst))
        output = self.fc_1(h_src)
        output = self.fc_2(output)
        return output


class FeatureProjector(keras.layers.Layer):
    def __init__(self, full_graph, itype, embedding_size):
        super().__init__()
        self.itype = itype
        self.full_graph = full_graph
        year_vocab_size = tf.reduce_max(full_graph.nodes[itype].data['year'])
        year_vocab_size = tf.cast(year_vocab_size, tf.int32) + 1
        genre_vocab_size = tf.shape(full_graph.nodes[itype].data['genre'])[1]
        id_vocab_size = full_graph.number_of_nodes(itype)
        self.year_embedding = keras.layers.Embedding(year_vocab_size, embedding_size)
        self.genre_embedding = keras.layers.Embedding(genre_vocab_size, embedding_size)
        self.id_embedding = keras.layers.Embedding(id_vocab_size, embedding_size)

    def call(self, induces_ids):
        year_ids = tf.nn.embedding_lookup(tf.cast(self.full_graph.nodes[self.itype].data['year'], tf.int32),
                                          induces_ids)  # None,
        year_ids = tf.expand_dims(year_ids, axis=0)  # 1,None
        #         print(year_ids.dtype)
        year_embedding = self.year_embedding(year_ids)  # 1,None,embedding
        year_embedding = tf.squeeze(year_embedding, axis=0)  # None,embedding

        genre = tf.nn.embedding_lookup(tf.cast(self.full_graph.nodes[self.itype].data['genre'], tf.int32), induces_ids)
        genre = tf.cast(genre, tf.int32)  # None,num_genres
        genre_embedding = self.genre_embedding(genre)  # None,num_genres,embedding
        genre_embedding = tf.reduce_mean(genre_embedding, axis=1)  # None,embedding

        item_ids = tf.nn.embedding_lookup(tf.cast(self.full_graph.nodes[self.itype].data['id'], tf.int32),
                                          induces_ids)  # None,
        item_ids = tf.expand_dims(item_ids, axis=0)  # 1,None
        id_embedding = self.id_embedding(item_ids)  # 1,None,embedding
        id_embedding = tf.squeeze(id_embedding, axis=0)  # None,embedding
        embedding = tf.concat([year_embedding, genre_embedding, id_embedding], axis=-1)
        return embedding
