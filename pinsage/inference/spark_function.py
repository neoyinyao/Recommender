import numpy as np
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import udf
from pyspark.sql.functions import pandas_udf


def split_array(df, col, size):
    for i in range(size):
        df = df.withColumn('feature{}'.format(i), df[col][i])
    return df


@pandas_udf(returnType=FloatType())
def multiply_func(a, b):
    return a * b


@pandas_udf(returnType=FloatType())
def divide_func(a, b):
    return a / b


@udf(returnType=ArrayType(FloatType()))
def concat(raw_feature, concat_feature):
    raw_feature.append(concat_feature)
    return raw_feature


@udf(returnType=ArrayType(FloatType()))
def l2norm(col):
    array = np.array(col)
    norm = np.linalg.norm(array)
    array = array / norm
    return array.tolist()


def fc(df, raw_col, weight, bias, new_col):
    """
    act like fully connect layer in DNN
    """

    @udf(returnType=ArrayType(FloatType()))
    def mlp_fn(x):
        return (np.dot(np.array(x), weight) + bias).tolist()

    df = df.withColumn(new_col, mlp_fn(raw_col))
    return df
