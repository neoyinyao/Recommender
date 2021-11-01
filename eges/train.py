import os
import argparse
import random

os.environ['DGLBACKEND'] = 'tensorflow'

from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

from model import DeepWalk, GES, EGES


@tf.function
def train_step(batch_data):
    inputs, labels = batch_data[:-1], batch_data[-1]
    with tf.GradientTape() as tape:
        logits = model(inputs)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)
        loss = tf.reduce_mean(loss)
    gradiens = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradiens, model.trainable_variables))
    return loss


def update_auc(query_embedding, pos_embedding, neg_embedding):
    pos_score = tf.reduce_sum(query_embedding * pos_embedding, axis=-1)  # None,1
    neg_score = tf.reduce_sum(query_embedding * neg_embedding, axis=-1)  # None,1
    pos_score = tf.nn.sigmoid(pos_score)
    neg_score = tf.nn.sigmoid(neg_score)
    pos_label = tf.ones_like(pos_score)
    neg_label = tf.zeros_like(pos_score)
    test_auc.update_state(pos_label, pos_score)
    test_auc.update_state(neg_label, neg_score)


@tf.function
def test_step_bge(batch_data):
    query_item_idxs, match_item_idxs, neg_item_idxs = batch_data
    query_embedding = model.get_hidden(query_item_idxs)
    pos_embedding = model.get_hidden(match_item_idxs)
    neg_embedding = model.get_hidden(neg_item_idxs)
    update_auc(query_embedding, pos_embedding, neg_embedding)


@tf.function
def test_step_ges_and_eges(batch_data):
    query_item_idxs, query_item_cat_idxs, query_item_brand_idxs, match_item_idxs, match_item_cat_idxs, \
    match_item_brand_idxs, neg_item_idxs, neg_item_cat_idxs, neg_item_brand_idxs = batch_data
    query_embedding = model.get_hidden((query_item_idxs, query_item_cat_idxs, query_item_brand_idxs))
    pos_embedding = model.get_hidden((match_item_idxs, match_item_cat_idxs, match_item_brand_idxs))
    neg_embedding = model.get_hidden((neg_item_idxs, neg_item_cat_idxs, neg_item_brand_idxs))
    update_auc(query_embedding, pos_embedding, neg_embedding)


if __name__ == '__main__':
    # load_metadata()
    parser = argparse.ArgumentParser(description="EGES model train config")
    parser.add_argument('--gpus', type=str, default='0', help='Which gpu to use.')
    parser.add_argument('--gpu_memory_limit', type=int, default=4096)
    parser.add_argument('--model_type', type=str, default='EGES', help='BGE or GES or EGES')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs.')
    parser.add_argument('--train_batch_size', type=int, default=1024, help='train batch size.')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='test batch size.')
    parser.add_argument('--auc_num_thresholds', type=int, default=10000, help='tensorflow approximate auc config')
    parser.add_argument('--test_steps', type=int, default=3000, help='test steps')
    parser.add_argument('--seed', type=int, default=4, help='random seed')
    args = parser.parse_args()
    SEED = args.seed
    random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate gpu_memory_limit of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.gpu_memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    from util import build_train_util
    from data_loader import build_dataset  # because import dgl automatic load all visible gpus, so import here

    train_g, test_item_pairs, item2idx, idx2item, train_item2cat, train_item2brand, brand_vocab, cat_vocab = build_train_util()
    sample_k, random_walk_length, num_ns = 3, 10, 5
    model_type = 'EGES'
    train_dataset = build_dataset(model_type, 'train', train_g, sample_k, random_walk_length, num_ns, cat_vocab,
                                  brand_vocab, train_item2cat,
                                  train_item2brand, item2idx, idx2item, test_item_pairs)
    test_dataset = build_dataset(model_type, 'test', train_g, sample_k, random_walk_length, num_ns, cat_vocab,
                                 brand_vocab, train_item2cat,
                                 train_item2brand, item2idx, idx2item, test_item_pairs)

    embedding_size = 160

    if model_type == 'BGE':
        model = DeepWalk(len(item2idx), embedding_size)
    elif model_type == 'GES':
        model = GES(len(item2idx), len(cat_vocab) + 1, len(brand_vocab) + 1, embedding_size)
    elif model_type == 'EGES':
        num_side = 2
        model = EGES(len(item2idx), len(cat_vocab) + 1, len(brand_vocab) + 1, embedding_size, num_side + 1)

    optimizer = tf.keras.optimizers.Adam()
    test_auc = keras.metrics.AUC()
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    prefetch_size = 10
    shuffle_ratio = 10
    train_dataset = train_dataset.shuffle(shuffle_ratio * train_batch_size).batch(train_batch_size).prefetch(
        prefetch_size)
    test_dataset = test_dataset.batch(test_batch_size).prefetch(prefetch_size)
    epochs = args.epochs
    test_steps = args.test_steps
    train_tqdm_util = tqdm(enumerate(train_dataset, start=1))
    for step, batch_data in train_tqdm_util:
        loss = train_step(batch_data)
        train_tqdm_util.set_description('step {0:d} loss {1:.4f}'.format(step, float(loss.numpy())))
        if step % test_steps == 0:
            for test_batch_data in tqdm(test_dataset):
                if model_type == 'BGE':
                    test_step_bge(test_batch_data)
                else:
                    test_step_ges_and_eges(test_batch_data)
            print("test step {0:d} , auc: {1:.4f}".format(step, float(test_auc.result().numpy())))
            test_auc.resetcle_states()
