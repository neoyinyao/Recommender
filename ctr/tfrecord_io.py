import random
import string
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm

num_int = 13
num_cat = 26
total_cols = 40
cat_imputation = [''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10)) for i in
                  range(num_cat)]  # random imputate null value


def build_vocab(train_file):
    cat_fea_count = {}
    with open(train_file) as f:
        for line in tqdm(f):
            line = line.split('\t')
            for i in range(num_int + 1, total_cols):
                if line[i] == '' or line[i] == '\n':
                    line[i] = cat_imputation[i - num_int - 1]
                if line[i] in cat_fea_count:
                    cat_fea_count[line[i]] += 1
                else:
                    cat_fea_count[line[i]] = 1

    cat_fea_vocab = {}
    idx = 0
    for key, count in cat_fea_count.items():
        if count > 10:
            cat_fea_vocab[key] = idx
            idx += 1
    with open('./data/cat_fea_vocab.pkl', 'wb') as f:
        pickle.dump(cat_fea_vocab, f)


def write_tfrecord(raw_file, output_file):
    with open('./data/cat_fea_vocab.pkl', 'rb') as f:
        cat_fea_vocab = pickle.load(f)
    with tf.io.TFRecordWriter(output_file) as writer:
        with open(raw_file) as f:
            for line in tqdm(f):
                line = line.split('\t')
                for i in range(1, num_int + 1):
                    if line[i] == '':  # continuous variable imputation by 0
                        line[i] = '0'
                    if int(line[i]) < 0:
                        line[i] = '0'
                int_features = list(map(int, line[1:1 + num_int]))
                int_array = np.array(int_features)
                int_array = int_array.astype(np.float32)
                int_array = np.log(int_array + 1)
                for i in range(num_int + 1, total_cols):
                    if line[i] == '' or line[i] == '\n':
                        line[i] = cat_imputation[i - num_int - 1]
                cat_features = [0 for i in range(num_cat)]
                cat_vals = line[num_int + 1:total_cols]
                for i in range(num_cat):
                    val = cat_vals[i]
                    if val in cat_fea_vocab:
                        cat_features[i] = cat_fea_vocab[val]
                    else:
                        cat_features[i] = 0
                cat_array = np.array(cat_features)
                label = int(line[0])
                feature = {
                    'int_features': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(int_array).numpy()])),
                    'cat_features': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(cat_array).numpy()])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())


def read_tfrecord(tfrecord_file):
    def _parse_example(example_string):  # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        feature_dict['int_features'] = tf.io.parse_tensor(feature_dict['int_features'], out_type=tf.float32)
        feature_dict['cat_features'] = tf.io.parse_tensor(feature_dict['cat_features'], out_type=tf.int64)
        return feature_dict

    feature_description = {  # 定义Feature结构，告诉解码器每个Feature的类型是什么
        'int_features': tf.io.FixedLenFeature([], tf.string),
        'cat_features': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)  # 读取 TFRecord 文件
    dataset = raw_dataset.map(_parse_example)
    dataset = dataset.map(lambda x: ({
                                         'int_features': x['int_features'],
                                         'cat_features': x['cat_features']
                                     }, x['label']))
    return dataset


if __name__ == '__main__':
    train_file = '../data/criteo/dac/train_split.txt'
    build_vocab(train_file)
    train_raw_file = '../data/criteo/dac/train_split.txt'
    train_output_file = '../data/criteo/dac/train_split.tfrecord'
    write_tfrecord(train_raw_file, train_output_file)
    test_raw_file = '../data/criteo/dac/test_split.txt'
    test_output_file = '../data/criteo/dac/test_split.tfrecord'
    write_tfrecord(test_raw_file, test_output_file)
