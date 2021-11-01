import os
import numpy as np
from pyspark.sql import SparkSession

from spark_function import fc, split_array, divide_func, multiply_func, l2norm, concat


def convolve(item_features_df, item_neighbors_df, conv_fc1_w, conv_fc1_b, conv_fc2_w, conv_fc2_b, convolve_hidden_size):
    """ sparksql realize PinSage Convolve dnn layer
    :param item_features_df:
    :param item_neighbors_df:
    :param conv_fc1_w:
    :param conv_fc1_b:
    :param conv_fc2_w:
    :param conv_fc2_b:
    :param convolve_hidden_size:
    :return:
    """
    # neighbor feature do dnn fully connect layer forward
    neighbor_features_df = item_features_df.withColumnRenamed('item_id', 'neighbor')
    neighbor_features_df = fc(neighbor_features_df, 'feature', conv_fc1_w, conv_fc1_b, 'feature')
    neighbor_features_df = split_array(neighbor_features_df, col='feature', size=convolve_hidden_size)
    neighbor_features_df = neighbor_features_df.drop('feature')

    # neighbor message update
    join_df = item_neighbors_df.join(neighbor_features_df, on='neighbor')
    for i in range(convolve_hidden_size):
        join_df = join_df.withColumn('feature{}'.format(str(i)), multiply_func('feature{}'.format(str(i)), 'weight'))
    join_df = join_df.groupBy('item_id').sum()
    for i in range(convolve_hidden_size):
        join_df = join_df.withColumn('feature{}'.format(str(i)), divide_func('sum(feature{})'.format(i), 'sum(weight)'))
    cols = ['item_id'] + ['feature{}'.format(i) for i in range(convolve_hidden_size)]
    join_df = join_df.select(cols)
    # concat neighbor feature and local feature,do dnn fully connect layer forward
    item_concat_df = join_df.join(item_features_df, on='item_id')
    for i in range(convolve_hidden_size):
        item_concat_df = item_concat_df.withColumn('feature', concat('feature', 'feature{}'.format(str(i))))
    item_concat_df = fc(item_concat_df, 'feature', conv_fc2_w, conv_fc2_b, 'feature')
    item_concat_df = item_concat_df.select('item_id', 'feature')
    item_concat_df = item_concat_df.withColumn('feature', l2norm('feature'))
    return item_concat_df


if __name__ == '__main__':
    os.environ["PYSPARK_PYTHON"] = '/home/kuer/anaconda3/envs/tf2.2/bin/python'  # set spark executor python path
    spark = SparkSession.builder.config("spark.executor.memory", "4g") \
        .config('spark.driver.memory', '4g') \
        .config('spark.executor.instances', '4') \
        .config('spark.sql.shuffle.partitions', '10') \
        .getOrCreate()  # config spark, create SparkSql context
    item_neighbors_df = spark.read.json('../data/item-neighbors.json')
    item_features_df = spark.read.json('../data/item-features.json')
    item_neighbors_df.printSchema()
    item_features_df.printSchema()
    embedding_size = len(item_features_df.head()['feature'])
    convolve_hidden_size, convolve_output_size = 32, 16
    conv1_fc1_w = np.random.normal(size=(embedding_size, convolve_hidden_size))
    conv1_fc1_b = np.random.normal(size=(convolve_hidden_size,))
    conv1_fc2_w = np.random.normal(size=(embedding_size + convolve_hidden_size, convolve_output_size))
    conv1_fc2_b = np.random.normal(size=(convolve_output_size,))
    conv2_fc1_w = np.random.normal(size=(convolve_output_size, convolve_hidden_size))
    conv2_fc1_b = np.random.normal(size=(convolve_hidden_size,))
    conv2_fc2_w = np.random.normal(size=(convolve_output_size + convolve_hidden_size, convolve_output_size))
    conv2_fc2_b = np.random.normal(size=(convolve_output_size,))
    item_conv1_df = convolve(item_features_df, item_neighbors_df, conv1_fc1_w, conv1_fc1_b, conv1_fc2_w, conv1_fc2_b,
                             convolve_hidden_size)
    item_conv2_df = convolve(item_conv1_df, item_neighbors_df, conv2_fc1_w, conv2_fc1_b, conv2_fc2_w, conv2_fc2_b,
                             convolve_hidden_size)
