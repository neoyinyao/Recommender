import os
import argparse

from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

from tfrecord_io import read_tfrecord
from base import BaseModel
from esmm import ESMM


def train_base(epochs, ctr_model, cvr_model, auc_num_thresholds, train_ctr_dataset, train_cvr_dataset,
               test_cvr_dataset, test_ctcvr_dataset):
    @tf.function
    def train_ctr_step(batch_data):
        feature, label = batch_data
        y_true = label[:, :1]  # None,
        with tf.GradientTape() as tape:
            y_pred = ctr_model(feature)  # None,1
            loss = keras.losses.binary_crossentropy(y_true, y_pred)
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, ctr_model.trainable_variables)
        ctr_optimizer.apply_gradients(zip(gradients, ctr_model.trainable_variables))
        train_ctr_auc.update_state(y_true, y_pred)
        return loss

    @tf.function
    def train_cvr_step(batch_data):
        feature, label = batch_data
        y_true = label[:, 1:]  # None,
        with tf.GradientTape() as tape:
            y_pred = cvr_model(feature)  # None,1
            loss = keras.losses.binary_crossentropy(y_true, y_pred)
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, cvr_model.trainable_variables)
        cvr_optimizer.apply_gradients(zip(gradients, cvr_model.trainable_variables))
        train_cvr_auc.update_state(y_true, y_pred)
        return loss

    @tf.function
    def test_cvr_step(batch_data):
        feature, label = batch_data
        y_true = label[:, 1:]  # None,
        y_pred = cvr_model(feature)  # None,1
        test_cvr_auc.update_state(y_true, y_pred)

    @tf.function
    def test_ctcvr_step(batch_data):
        feature, label = batch_data
        y_true = label[:, 1:]  # None,
        y_pred = ctr_model(feature) * cvr_model(feature)  # None,1
        test_ctcvr_auc.update_state(y_true, y_pred)

    ctr_optimizer = tf.keras.optimizers.Adam()
    cvr_optimizer = tf.keras.optimizers.Adam()
    train_ctr_auc = tf.keras.metrics.AUC(num_thresholds=auc_num_thresholds)
    train_cvr_auc = tf.keras.metrics.AUC(num_thresholds=auc_num_thresholds)
    test_cvr_auc = tf.keras.metrics.AUC(num_thresholds=auc_num_thresholds)
    test_ctcvr_auc = tf.keras.metrics.AUC(num_thresholds=auc_num_thresholds)
    for epoch in range(1, epochs + 1):
        train_ctr_util = tqdm(train_ctr_dataset)
        train_cvr_util = tqdm(train_cvr_dataset)
        print('epoch {} train ctr'.format(epoch))
        for step, batch_data in enumerate(train_ctr_util, start=1):  # use impression data
            loss = train_ctr_step(batch_data)
            train_ctr_util.set_description(
                'train epoch {0:d} step {1:d} loss {2:.4f} ctr_auc {3:.4f}'.format(epoch, step, float(loss.numpy()),
                                                                                   float(
                                                                                       train_ctr_auc.result().numpy())))
        print('epoch {} train cvr'.format(epoch))
        for step, batch_data in enumerate(train_cvr_util, start=1):  # use clicked data
            loss = train_cvr_step(batch_data)
            train_cvr_util.set_description(
                'train epoch {0:d} step {1:d} loss {2:.4f} cvr_auc {3:.4f}'.format(epoch, step, float(loss.numpy()),
                                                                                   float(
                                                                                       train_cvr_auc.result().numpy())))
        print('\n')
        for batch_data in tqdm(test_cvr_dataset):  # use clicked data
            test_cvr_step(batch_data)
        print('test epoch {0:d} cvr_auc {1:.4f} '.format(epoch, float(test_cvr_auc.result().numpy())))

        for batch_data in tqdm(test_ctcvr_dataset):
            test_ctcvr_step(batch_data)
        print('test epoch {0:d} ctcvr_auc {1:.4f} '.format(epoch, float(test_ctcvr_auc.result().numpy())))
        print('\n')
        train_ctr_auc.reset_states()
        train_cvr_auc.reset_states()
        test_cvr_auc.reset_states()
        test_ctcvr_auc.reset_states()


def train_esmm(epochs, model, auc_num_thresholds, train_ctr_dataset, test_cvr_dataset,
               test_ctcvr_dataset, test_steps):
    @tf.function
    def train_step(batch_data):
        feature, label = batch_data
        with tf.GradientTape() as tape:
            y_pred = model(feature)
            loss = tf.keras.losses.binary_crossentropy(label, y_pred, from_logits=False)
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        y_true_ctcvr = label[:, 1]  # None,
        y_pred_ctcvr = y_pred[:, 1]  # None,1
        train_ctcvr_auc.update_state(y_true_ctcvr, y_pred_ctcvr)
        return loss

    @tf.function
    def test_step_cvr(batch_data):
        feature, label = batch_data
        y_true = label[:, 1]
        y_pred = model.compute_cvr(feature)
        test_cvr_auc.update_state(y_true, y_pred)

    @tf.function
    def test_step_ctcvr(batch_data):
        feature, label = batch_data
        y_true = label[:, 1]
        y_pred = model.compute_ctcvr(feature)
        test_ctcvr_auc.update_state(y_true, y_pred)

    optimizer = tf.keras.optimizers.Adam()
    epochs = epochs
    train_ctcvr_auc = tf.keras.metrics.AUC(num_thresholds=auc_num_thresholds)
    test_cvr_auc = tf.keras.metrics.AUC(num_thresholds=auc_num_thresholds)
    test_ctcvr_auc = tf.keras.metrics.AUC(num_thresholds=auc_num_thresholds)
    for epoch in range(1, epochs + 1):
        tqdm_util = tqdm(train_ctr_dataset)
        for step, batch_data in enumerate(tqdm_util, start=1):
            loss = train_step(batch_data)
            tqdm_util.set_description(
                'train epoch {0:d} step {1:d} loss {2:.4f} ctcvr_auc {3:.4f}'.format(epoch, step, float(loss.numpy()),
                                                                                     train_ctcvr_auc.result().numpy()))
            if step % test_steps == 0:
                for test_batch_data in tqdm(test_cvr_dataset):
                    test_step_cvr(test_batch_data)
                print('test epoch {0:d} step {1:d} cvr_auc {2:.4f}: '.format(epoch, step,
                                                                             float(test_cvr_auc.result().numpy())))
                for test_batch_data in tqdm(test_ctcvr_dataset):
                    test_step_ctcvr(test_batch_data)
                print(
                    'test epoch {0:d} step {1:d} ctcvr_auc {2:.4f}: '.format(epoch, step,
                                                                             float(test_ctcvr_auc.result().numpy())))
                print('\n')
                test_cvr_auc.reset_states()
                test_ctcvr_auc.reset_states()
        train_ctcvr_auc.reset_states()


def train():
    parser = argparse.ArgumentParser(description="ESMM model train config")
    parser.add_argument('--gpus', type=str, default='0',
                        help='Which gpu to use.')
    parser.add_argument('--gpu_memory_limit', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs.')
    parser.add_argument('--train_batch_size', type=int, default=512,
                        help='train batch size.')
    parser.add_argument('--test_batch_size', type=int, default=2048,
                        help='test batch size.')
    parser.add_argument('--auc_num_thresholds', type=int, default=10000,
                        help='tensorflow approximate auc config')
    parser.add_argument('--train_ctr_tfrecord', type=str, default='data/train_impression_subsample.tfrecord',
                        help='train tfrecord path')
    parser.add_argument('--model_type', type=str, default='ESMM',
                        help='BASE or ESMM')
    parser.add_argument('--test_steps', type=int, default=3000,
                        help='test steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    args = parser.parse_args()
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
    seed = args.seed
    tf.random.set_seed(seed)
    epochs = args.epochs
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    auc_num_thresholds = args.auc_num_thresholds
    train_ctr_tfrecord = args.train_ctr_tfrecord
    model_type = args.model_type
    test_steps = args.test_steps
    feat_vocab = {
        '101': 238635,
        '121': 98,
        '122': 14,
        '124': 3,
        '125': 8,
        '126': 4,
        '127': 4,
        '128': 3,
        '129': 5,
        '205': 467298,
        '206': 6929,
        '207': 263942,
        '216': 106399,
        '508': 5888,
        '509': 104830,
        '702': 51878,
        '853': 37148,
        '301': 4}
    embedding_size = 18
    mlp_units = [360, 200, 80, 1]
    prefetch_size = 10
    train_ctr_dataset = read_tfrecord(train_ctr_tfrecord)
    train_ctr_dataset = train_ctr_dataset.shuffle(100 * train_batch_size).batch(
        train_batch_size).prefetch(prefetch_size)

    train_cvr_tfrecord = 'data/train_click.tfrecord'
    train_cvr_dataset = read_tfrecord(train_cvr_tfrecord)
    train_cvr_dataset = train_cvr_dataset.shuffle(100 * train_batch_size).batch(train_batch_size).prefetch(
        prefetch_size)

    test_cvr_tfrecord = 'data/test_click.tfrecord'
    test_cvr_dataset = read_tfrecord(test_cvr_tfrecord)
    test_cvr_dataset = test_cvr_dataset.batch(test_batch_size).prefetch(prefetch_size)

    test_ctcvr_tfrecord = 'data/test_impression.tfrecord'
    test_ctcvr_dataset = read_tfrecord(test_ctcvr_tfrecord)
    test_ctcvr_dataset = test_ctcvr_dataset.batch(test_batch_size).prefetch(prefetch_size)
    if model_type == 'BASE':
        last_activation = 'sigmoid'
        ctr_model = BaseModel(mlp_units, last_activation, feat_vocab, embedding_size)
        cvr_model = BaseModel(mlp_units, last_activation, feat_vocab, embedding_size)
        train_base(epochs, ctr_model, cvr_model, auc_num_thresholds, train_ctr_dataset, train_cvr_dataset,
                   test_cvr_dataset, test_ctcvr_dataset)
    elif model_type == 'ESMM':
        model = ESMM(mlp_units, feat_vocab, embedding_size)
        train_esmm(epochs, model, auc_num_thresholds, train_ctr_dataset, test_cvr_dataset, test_ctcvr_dataset,
                   test_steps)


if __name__ == '__main__':
    train()
