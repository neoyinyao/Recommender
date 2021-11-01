import os
import pickle

os.environ['DGLBACKEND'] = 'tensorflow'

import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import numpy as np
from tensorflow.keras.metrics import Mean

from model import PinSageModel
from data_loader import PinSageSampler, item2item_batch_sampler
from evaluation import get_item_reprs, recommend, hit_rate_eval


def margin_loss(pos_score, neg_score, delta):
    loss = tf.clip_by_value(neg_score + delta - pos_score, 0, np.inf)
    loss = tf.reduce_mean(loss)
    return loss


def negative_log_likelihood_loss(pos_score, neg_score):
    """
    :param pos_score: None,1
    :param neg_score: None,1
    :return:
    """
    score = tf.concat([pos_score, neg_score], axis=0)
    score = tf.squeeze(score, axis=-1)
    pos_label = tf.ones_like(pos_score)
    neg_label = tf.zeros_like(neg_score)
    label = tf.concat([pos_label, neg_label], axis=0)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(label, score)
    loss = tf.reduce_mean(loss)
    return loss


# TODO :@tf.function, dgl in tensorflow static computation graph raise error,only run in eager mode
def train_step(batch_data):
    with tf.GradientTape() as tape:
        pos_graph, neg_graph, blocks = batch_data
        pos_score, neg_score = model(pos_graph, neg_graph, blocks)
        loss = margin_loss(pos_score, neg_score, delta=1.)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_metric.update_state(loss)
    return loss


if __name__ == '__main__':
    with open('../data/train_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    train_g = dataset['train-graph']
    val_matrix = dataset['val-matrix']
    test_matrix = dataset['test-matrix']
    utype = dataset['user-type']
    itype = dataset['item-type']
    u2i_etype = dataset['user-to-item-type']
    i2u_etype = dataset['item-to-user-type']
    timestamp = dataset['timestamp-edge-column']

    num_layers = 2
    embedding_size = 8
    conv_hidden_size, conv_output_size = 32, 16
    train_batch_size = 32
    test_batch_size = 32
    top_k = 10
    random_walk_length, num_random_walks, termination_prob, num_neighbors = 2, 4, 0, 3
    test_steps = 1000
    train_g.nodes[itype].data['id'] = tf.constant(np.arange(train_g.number_of_nodes(itype)))
    model = PinSageModel(train_g, itype, num_layers, embedding_size, conv_hidden_size, conv_output_size)
    optimizer = keras.optimizers.Adam()
    sampler = PinSageSampler(train_g, itype, utype, num_layers, random_walk_length, num_random_walks,
                             termination_prob, num_neighbors)
    item2item_generator = item2item_batch_sampler(train_g, utype, itype, train_batch_size)
    train_metric = Mean()
    tqdm_util = tqdm(item2item_generator)
    for step, (heads, pos_tails, neg_tails) in enumerate(tqdm_util, start=0):
        batch_data = sampler.sample_from_item_pairs(heads, pos_tails, neg_tails, itype)
        loss = train_step(batch_data)
        tqdm_util.set_description(
            'step {0:d} step_loss {1:.4f} '.format(step, float(loss.numpy())))
        if step % test_steps == 0:
            item_reprs = get_item_reprs(model, sampler, train_g, itype, test_batch_size)
            recommends = recommend(train_g, top_k, item_reprs, u2i_etype, utype, timestamp, test_batch_size)
            hit_rate = hit_rate_eval(recommends, val_matrix.tocsr())
            print('step {0:d} hit_tate {1:.4f}'.format(step, hit_rate))
            # tqdm_util.write('step {0:d} hit_tate {1:.4f} \n'.format(step, hit_rate))
            # tqdm.write('step {0:d} hit_tate {1:.4f}'.format(step, hit_rate))
