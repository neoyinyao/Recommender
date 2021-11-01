from tqdm import tqdm
import tensorflow as tf

cols = [
    '101',
    '121',
    '122',
    '124',
    '125',
    '126',
    '127',
    '128',
    '129',
    '205',
    '206',
    '207',
    '216',
    '508',
    '509',
    '702',
    '853',
    '301']


def write_impression_tfrecord(raw_file, tfrecord_file):
    """
            size        click       conversion
    train   42299905    1644256     8802
    test    43016614    1673447     9195
    :param tfrecord_file:
    :param raw_file:
    :return:
    """
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        with open(raw_file) as f:
            f.readline()  # read header, filter first line
            for line in tqdm(f):
                line = line.strip().split(',')
                line = list(map(int, line))
                label = line[0:2]
                feature = line[2:]
                label_feature = {'label': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(label).numpy()]))}
                feature = {
                    cols[i]: tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor([feature[i]]).numpy()])) for i in
                    range(len(cols))
                }
                feature.update(label_feature)
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())


def write_impression_tfrecord_with_subsample(raw_file, tfrecord_file, subsample_ratio=5):
    """
    sample non_click example, balance dataset, after click:non_click = 1:5
    :param tfrecord_file:
    :param raw_file:
    :param subsample_ratio:non_click_num : click_num
    :return:
    """
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        with open(raw_file) as f:
            f.readline()  # read header, filter first line
            non_click_count = 0
            for line in tqdm(f):
                line = line.strip().split(',')
                line = list(map(int, line))
                if line[0] == 0:
                    non_click_count += 1
                if line[0] == 0 and non_click_count % subsample_ratio != 0:
                    continue
                label = line[0:2]
                feature = line[2:]
                label_feature = {'label': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(label).numpy()]))}
                feature = {
                    cols[i]: tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor([feature[i]]).numpy()]))
                    for i in range(len(cols))
                }
                feature.update(label_feature)
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())


def write_click_tfrecord(raw_file, tfrecord_file):
    """
    sample non_click example, balance dataset
    :param tfrecord_file:
    :param raw_file:
    :return:
    """
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        with open(raw_file) as f:
            f.readline()  # read header, filter first line
            for line in tqdm(f):
                line = line.strip().split(',')
                line = list(map(int, line))
                if line[0] == 1:
                    label = line[0:2]
                    feature = line[2:]
                    label_feature = {'label': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(label).numpy()]))}
                    feature = {
                        cols[i]: tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor([feature[i]]).numpy()])) for i
                        in
                        range(len(cols))
                    }
                    feature.update(label_feature)
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())


def read_tfrecord(tfrecord_file):
    feature_description = {  # 定义Feature结构，告诉解码器每个Feature的类型是什么
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    add_feature_description = {
        col: tf.io.FixedLenFeature([], tf.string) for col in cols
    }
    feature_description.update(add_feature_description)

    def _parse_example(example_string):  # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        feature_dict['label'] = tf.io.parse_tensor(feature_dict['label'], out_type=tf.int32)
        for col in cols:
            feature_dict[col] = tf.io.parse_tensor(feature_dict[col], out_type=tf.int32)
        return feature_dict

    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(_parse_example)
    feature_dataset = dataset.map(lambda x: {feat: tf.reshape(x[feat], (-1,)) for feat in cols})
    label_dataset = dataset.map(lambda x: x['label'])
    label_dataset = label_dataset.map(lambda x: tf.reshape(x, (2,)))
    zip_dataset = tf.data.Dataset.zip((feature_dataset, label_dataset))
    return zip_dataset


if __name__ == '__main__':
    train_file = 'data/ctr_cvr.train'
    test_file = 'data/ctr_cvr.test'

    train_tfrecord = 'data/train_impression.tfrecord'
    write_impression_tfrecord(train_file, train_tfrecord)
    test_tfrecord = 'data/test_impression.tfrecord'
    write_impression_tfrecord(test_file, test_tfrecord)

    train_subsample_tfrecord = 'data/train_impression_subsample.tfrecord'
    write_impression_tfrecord_with_subsample(train_file, train_subsample_tfrecord)

    train_click_tfrecord = 'data/train_click.tfrecord'
    write_click_tfrecord(train_click_tfrecord, train_file)
    test_click_tfrecord = 'data/test_click.tfrecord'
    write_click_tfrecord(test_click_tfrecord, test_file)
