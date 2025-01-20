# encoding=utf-8
import json
from collections import OrderedDict
# import os,sys
# currentPath = os.path.split(os.path.realpath(__file__))[0]
# sys.path.append(currentPath + os.sep + '../')
# sys.path.append(currentPath + os.sep + '../..')
from fg.feature_generator_seq import FeatureGenerator
from tensorflow.contrib import layers
import tensorflow as tf

def fg_mid(global_conf, fg_conf):
    """
    :param global_conf: 全局配置文件,json格式
    :param fg_conf: 特征生成器配置文件,路径
    :return: block_column_dict: {block1: [feature_column1, feature_column2], block2: [feature_column3, feature_column4]}

    """
    # [feature_name1, feature_name2]
    feature_names = []
    print("parsing following blocks:")
    for block in global_conf["input_columns"]:
        # 通过+的方式将所有的feature_name拼接成一个list
        feature_names += global_conf["input_columns"][block]
    print("feature_names: ", feature_names)
    # 从fg_conf里边挑出来feature_names对应的feature_conf，并且生成feature_columns
    fg = FeatureGenerator(fg_conf, feature_names, global_conf["model"])
    block_column_dict = {}
    seq_block_column_dict = OrderedDict()
    for block in global_conf["input_columns"]:
        # 如果是sequence的block，那么就需要生成sequence的block_column
        if block in fg.sequence_names:
            seq_block_column_dict[block] = OrderedDict()
            feature_names = [
                "{}_{}".format(block, i) for i in global_conf["input_columns"][block]
            ]
            for feature_name in feature_names:
                seq_block_column_dict[block][
                    feature_name
                ] = fg.seq_feature_columns_from_name([feature_name])
        else:
            # 一次取出一个block的所有feature_names，然后生成block_column
            feature_names = global_conf["input_columns"][block]
            # 如果这里报错，那么就是fg.json里边没有global对应的feature_name,请去fg.json里边添加，或者检查/删除global里边的feature_name
            block_column = fg.feature_columns_from_name(feature_names)
            block_column_dict[block] = block_column
    return fg, block_column_dict, seq_block_column_dict, fg._feature_columns


def get_sequence_layer(
    features, seq_column_dict, seq_length_column, max_seq_len, scope="sequence_layer"
):
    layer_dict = OrderedDict()
    # [1024 1]
    seq_length_layer = layers.input_from_feature_columns(
        features, seq_length_column, scope=scope
    )
    for name, column in seq_column_dict.items():
        # [51200 8] 51200=50(max sequence len)*1024(batchsize)
        layer = layers.input_from_feature_columns(features, column, scope=scope)
        # [50 1024 8]
        layer = tf.split(layer, max_seq_len, axis=0)  # [?, f_num*f_embedding]*30
        # [1024 50 8]
        layer = tf.stack(values=layer, axis=1)

        max_length, dim = layer.get_shape().as_list()[1:3]
        # [1024 1 50]
        masks = tf.sequence_mask(seq_length_layer, max_length)

        # [1024*50,8]
        # use tf.zeros_like(items_2d) to mask
        # use masks as condition to choose
        items_2d = tf.reshape(layer, [-1, tf.shape(layer)[2]])
        layer = tf.reshape(
            tf.where(tf.reshape(masks, [-1]), items_2d, tf.zeros_like(items_2d)),
            tf.shape(layer),
        )
        # 1024 50 8
        layer_dict[name] = layer
    return layer_dict, seq_length_layer


# 负责将rtp_fg之后的特征user_trigger_seq_X_cate_id转换为user_trigger_seq_cate_id
def sequence_features(original_features, feature_names, length, sequence_name):
    features_list_tensor = OrderedDict()
    sequence_features_tensor = OrderedDict()
    for feature_name in feature_names:
        feature_tensor_list = []
        for i in range(length):
            feature_name_in_tensor = "{}_{}_{}".format(sequence_name, i, feature_name)
            feature_tensor_list.append(original_features[feature_name_in_tensor])
        features_list_tensor[feature_name] = feature_tensor_list
    for feature_name, tensor_list in features_list_tensor.items():
        try:
            sequence_features_tensor[
                sequence_name + "_" + feature_name
            ] = tf.sparse_concat(
                sp_inputs=tensor_list, axis=0, expand_nonconcat_dim=True
            )
        except:
            sequence_features_tensor[sequence_name + "_" + feature_name] = tf.concat(
                values=tensor_list, axis=0
            )
    return sequence_features_tensor


def concat_seq_features(fg, global_conf, features):
    seq_list = []
    seq_length_list_dcit = {}
    for value in fg.get_feature_conf.values():
        for v in value:
            if "sequence_name" in v:
                seq_list.append(v["sequence_name"])
                seq_length_list_dcit[v["sequence_name"]] = v["sequence_length"]

    for block_name in global_conf["input_columns"]:
        if block_name in seq_list:
            column_list = global_conf["input_columns"][block_name]
            for column in column_list:
                features.update(
                    sequence_features(
                        features, [column], seq_length_list_dcit[block_name], block_name
                    )
                )
def get_feature(features, block_column_dict, block_name,Print=False,BN=False,is_training=True,scope=""):
    feature_column = block_column_dict[block_name]
    # 如果这里不加scope，当传进去的scope为none时，会自动新建一个input_from_feature_columns的scope
    feature = layers.input_from_feature_columns(features, feature_column, scope=scope)
    if BN:
        feature = layers.batch_norm(feature, scale=True, is_training=is_training)
    column_dimension = []
    for column in sorted(set(feature_column), key=lambda x: x.key):
        column_dimension.append([column.key, column.dimension])
    return feature
def get_column_dimension(block_column_dict,block_name):
    feature_column = block_column_dict[block_name]
    column_dimension = {}
    dimension = 0
    for column in sorted(set(feature_column), key=lambda x: x.key):
        column_dimension[column] = [dimension, dimension + column.dimension]
        dimension += column.dimension
    return column_dimension
def get_seq_feature(features, seq_block_column_dict, block_column_dict,seq_max_length,seq_block='hw_click_seq',Print=False,scope="get_seq_feature"):
    seq_column_dict = seq_block_column_dict[seq_block]
    seq_length_column = block_column_dict[seq_block + "_length"]
    seq_layer_dict, seq_length_layer = get_sequence_layer(
        features, seq_column_dict, seq_length_column, seq_max_length, scope=scope
    )
    seq_layer_inputs = tf.concat(
        [item[1] for item in seq_layer_dict.items()], -1
    )
    return seq_layer_inputs,seq_length_layer

if __name__ == "__main__":
    global_conf = json.loads(open("../conf/find_similar_rank/global_conf_seq.json").read())
    fg_conf = "../conf/find_similar_rank/fg_seq.json"
    fg, result1, result2, result3 = fg_mid(global_conf, fg_conf)
    print(result1)
    print(result2)
