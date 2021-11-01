import os
import argparse
import random

import tensorflow as tf

from model import DLRM, DeepFM
from tfrecord_io import read_tfrecord


def debug():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--model_type', type=str, default='DLRM')
    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=4)
    args = parser.parse_args()
    seed = args.seed
    tf.random.set_seed(seed)
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    model_type = args.model_type
    train_tfrecord_file = './data/train_split.tfrecord'
    test_tfrecord_file = './data/test_split.tfrecord'
    prefetch_ratio = 10
    train_dataset = read_tfrecord(train_tfrecord_file).shuffle(100).prefetch(train_batch_size * 10).batch(
        train_batch_size).take(10)
    test_dataset = read_tfrecord(test_tfrecord_file).prefetch(prefetch_ratio * test_batch_size).batch(
        test_batch_size).take(10)
    num_int_fea = 13
    num_cat_fea = 26
    vocab_size = 1000000
    embedding_size = 16
    epochs = 10
    log_path = './logs'
    ckpt_path = './ckpts'
    log_dir = os.path.join(log_path, model_type)
    checkpoint_path = os.path.join(ckpt_path, model_type, 'checkpoint')
    if model_type == 'DLRM':
        bottom_mlp_units = [512, 256, 64, 16]
        top_mlp_units = [512, 256, 1]
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = DLRM(bottom_mlp_units, top_mlp_units, embedding_size, vocab_size, num_cat_fea, num_int_fea)
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            optimizer = tf.keras.optimizers.Adam()
            metrics = [tf.keras.metrics.AUC(), tf.keras.metrics.BinaryAccuracy()]
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                monitor='val_auc',
                mode='max',
                save_best_only=True)
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir),
                     tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_auc', mode='max'),
                     model_checkpoint_callback]
        model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=callbacks)
    elif model_type == 'DeepFM':
        strategy = tf.distribute.MirroredStrategy()
        mlp_units = [512, 256, 1]
        with strategy.scope():
            model = DeepFM(embedding_size, vocab_size, num_int_fea, num_cat_fea, mlp_units)
            loss_fn = tf.keras.losses.BinaryCrossentropy()
            optimizer = tf.keras.optimizers.Adam()
            metrics = [tf.keras.metrics.AUC(), tf.keras.metrics.BinaryAccuracy()]
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                monitor='val_auc',
                mode='max',
                save_best_only=True)
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir),
                     tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_auc', mode='max'),
                     model_checkpoint_callback]
        model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=callbacks)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--gpu_memory_limit', type=int, default=4096)
    parser.add_argument('--model_type', type=str, default='DLRM')
    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=4)
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    tf.random.set_seed(seed)
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
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    model_type = args.model_type
    train_tfrecord_file = './data/train_split.tfrecord'
    test_tfrecord_file = './data/test_split.tfrecord'
    prefetch_num = 10
    train_dataset = read_tfrecord(train_tfrecord_file).shuffle(100).batch(
        train_batch_size).prefetch(prefetch_num)
    test_dataset = read_tfrecord(test_tfrecord_file).batch(test_batch_size).prefetch(prefetch_num)
    num_int_fea = 13
    num_cat_fea = 26
    vocab_size = 1000000
    embedding_size = 16
    epochs = 3
    log_path = './logs'
    ckpt_path = './ckpts'
    log_dir = os.path.join(log_path, model_type)
    checkpoint_path = os.path.join(ckpt_path, model_type, 'checkpoint')
    if model_type == 'DLRM':
        bottom_mlp_units = [512, 256, 64, 16]
        top_mlp_units = [512, 256, 1]
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = DLRM(bottom_mlp_units, top_mlp_units, embedding_size, vocab_size, num_cat_fea, num_int_fea)
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            optimizer = tf.keras.optimizers.Adam()
            metrics = [tf.keras.metrics.AUC(), tf.keras.metrics.BinaryAccuracy()]
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                monitor='val_auc',
                mode='max',
                save_best_only=True)
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir),
                     tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_auc', mode='max'),
                     model_checkpoint_callback]
        model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=callbacks)
    elif model_type == 'DeepFM':
        strategy = tf.distribute.MirroredStrategy()
        mlp_units = [512, 256, 1]
        with strategy.scope():
            model = DeepFM(embedding_size, vocab_size, num_int_fea, num_cat_fea, mlp_units)
            loss_fn = tf.keras.losses.BinaryCrossentropy()
            optimizer = tf.keras.optimizers.Adam()
            metrics = [tf.keras.metrics.AUC(), tf.keras.metrics.BinaryAccuracy()]
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                monitor='val_auc',
                mode='max',
                save_best_only=True)
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir),
                     tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_auc', mode='max'),
                     model_checkpoint_callback]
        model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=callbacks)


if __name__ == '__main__':
    train()
