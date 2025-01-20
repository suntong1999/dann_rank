import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope

from model_ops import ops as base_ops


def tower(input, name, dnn_l2_reg, is_training, dnn_hidden_units, need_dropout, dropout_rate):
    net = input
    with tf.variable_scope(name_or_scope="%s-DNN" % name):
        with arg_scope(base_ops.model_arg_scope(weight_decay=dnn_l2_reg)):
            for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                with tf.variable_scope("Hidden_Layer_%d" % layer_id) as dnn_hidden_layer_scope:
                    net = layers.fully_connected(net, num_hidden_units, activation_fn=tf.tanh,
                                                 scope=dnn_hidden_layer_scope, normalizer_fn=layers.batch_norm,
                                                 normalizer_params={"scale": True, "is_training": is_training, })
                    if need_dropout:
                        net = tf.layers.dropout(net, rate=dropout_rate, training=is_training)

            with tf.variable_scope(name_or_scope="Logits") as deep_logits_scope:
                with arg_scope(base_ops.model_arg_scope(weight_decay=dnn_l2_reg)):
                    concat_layer = tf.concat([net], axis=1)
                    logits = layers.fully_connected(concat_layer, 1, scope=deep_logits_scope)

    return net, logits


def tower2(input, name, dnn_l2_reg, is_training, dnn_hidden_units, need_dropout, dropout_rate):
    nets = []
    net = input
    with tf.variable_scope(name_or_scope="%s-DNN" % name):
        with arg_scope(base_ops.model_arg_scope(weight_decay=dnn_l2_reg)):
            for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                with tf.variable_scope("Hidden_Layer_%d" % layer_id) as dnn_hidden_layer_scope:
                    net = layers.fully_connected(net, num_hidden_units, activation_fn=tf.tanh,
                                                 variables_collections=["collections_dnn_hidden_layer"],
                                                 outputs_collections=["collections_dnn_hidden_output"],
                                                 scope=dnn_hidden_layer_scope,
                                                 normalizer_fn=layers.batch_norm,
                                                 normalizer_params={"scale": True, "is_training": is_training})
                    if need_dropout:
                        net = tf.layers.dropout(net, rate=dropout_rate, noise_shape=None, seed=None,
                                                training=is_training, name=None)
                    nets.append(net)

            with tf.variable_scope(name_or_scope="Logits") as deep_logits_scope:
                with arg_scope(base_ops.model_arg_scope(weight_decay=dnn_l2_reg)):
                    concat_layer = tf.concat([net], axis=1)
                    logits = layers.fully_connected(concat_layer, 1, activation_fn=None,
                                                    variables_collections=["collections_dnn_hidden_layer"],
                                                    outputs_collections=["collections_dnn_hidden_output"],
                                                    scope=deep_logits_scope)

    return nets, logits

def tower_adapter(input, adapters, name, dnn_l2_reg, is_training, dnn_hidden_units, need_dropout, dropout_rate):
    nets = []
    net = input
    with tf.variable_scope(name_or_scope="%s-DNN" % name):
        with arg_scope(base_ops.model_arg_scope(weight_decay=dnn_l2_reg)):
            for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                with tf.variable_scope("Hidden_Layer_%d" % layer_id) as dnn_hidden_layer_scope:
                    net = layers.fully_connected(net, num_hidden_units, activation_fn=tf.tanh,
                                                 variables_collections=["collections_dnn_hidden_layer"],
                                                 outputs_collections=["collections_dnn_hidden_output"],
                                                 scope=dnn_hidden_layer_scope,
                                                 normalizer_fn=layers.batch_norm,
                                                 normalizer_params={"scale": True, "is_training": is_training})
                    if need_dropout:
                        net = tf.layers.dropout(net, rate=dropout_rate, noise_shape=None, seed=None,
                                                training=is_training, name=None)
                    net = net * adapters[layer_id]
                    nets.append(net)

            with tf.variable_scope(name_or_scope="Logits") as deep_logits_scope:
                with arg_scope(base_ops.model_arg_scope(weight_decay=dnn_l2_reg)):
                    concat_layer = tf.concat([net], axis=1)
                    logits = layers.fully_connected(concat_layer, 1, activation_fn=None,
                                                    variables_collections=["collections_dnn_hidden_layer"],
                                                    outputs_collections=["collections_dnn_hidden_output"],
                                                    scope=deep_logits_scope)

    return nets, logits
def lhuc_tower(input, lhuc_input, name, dnn_l2_reg, is_training, dnn_hidden_units, need_dropout, dropout_rate):
    net = input
    with tf.variable_scope(name_or_scope="%s-DNN" % name):
        with arg_scope(base_ops.model_arg_scope(weight_decay=dnn_l2_reg)):
            for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                # tower FC
                with tf.variable_scope("Hidden_Layer_%d" % layer_id) as dnn_hidden_layer_scope:
                    net = layers.fully_connected(net, num_hidden_units, activation_fn=tf.tanh,

                                                 scope=dnn_hidden_layer_scope, normalizer_fn=layers.batch_norm,
                                                 normalizer_params={"scale": True,
                                                                    "is_training": is_training})
                # bias layer activation
                with tf.variable_scope("lhuc_output_layer_%d" % layer_id) as lhuc_scale_layer_scope:
                    lhuc_output = layers.fully_connected(lhuc_input, 64, activation_fn=tf.nn.relu,

                                                         scope=lhuc_scale_layer_scope, normalizer_fn=layers.batch_norm,
                                                         normalizer_params={"scale": True,
                                                                            "is_training": is_training})
                # lhuc scale, same shape as FC
                with tf.variable_scope("lhuc_scale_layer_%d" % layer_id) as lhuc_scale_layer_scope:
                    lhuc_scale = layers.fully_connected(lhuc_output, int(net.shape[1]), activation_fn=tf.sigmoid,
                                                        scope=lhuc_scale_layer_scope)
                    net = net * lhuc_scale * 2.0
                if need_dropout:
                    net = tf.layers.dropout(net, rate=dropout_rate, noise_shape=None, seed=None,
                                            training=is_training, name=None)
            # record last layer tenor
            last_net = net
            with tf.variable_scope(name_or_scope="Logits") as deep_logits_scope:
                with arg_scope(base_ops.model_arg_scope(weight_decay=dnn_l2_reg)):
                    concat_layer = tf.concat([net], axis=1)
                    logits = layers.fully_connected(concat_layer, 1, activation_fn=None,
                                                    scope=deep_logits_scope)
    return net, logits
def lhuc_tower_tanh(input, lhuc_input, name, dnn_l2_reg, is_training, dnn_hidden_units, need_dropout, dropout_rate):
    net = input
    with tf.variable_scope(name_or_scope="%s-DNN" % name):
        with arg_scope(base_ops.model_arg_scope(weight_decay=dnn_l2_reg)):
            for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                # tower FC
                with tf.variable_scope("Hidden_Layer_%d" % layer_id) as dnn_hidden_layer_scope:
                    net = layers.fully_connected(net, num_hidden_units, activation_fn=tf.tanh,

                                                 scope=dnn_hidden_layer_scope, normalizer_fn=layers.batch_norm,
                                                 normalizer_params={"scale": True,
                                                                    "is_training": is_training})
                # bias layer activation
                with tf.variable_scope("lhuc_output_layer_%d" % layer_id) as lhuc_scale_layer_scope:
                    lhuc_output = layers.fully_connected(lhuc_input, 64, activation_fn=tf.tanh,

                                                         scope=lhuc_scale_layer_scope, normalizer_fn=layers.batch_norm,
                                                         normalizer_params={"scale": True,
                                                                            "is_training": is_training})
                # lhuc scale, same shape as FC
                with tf.variable_scope("lhuc_scale_layer_%d" % layer_id) as lhuc_scale_layer_scope:
                    lhuc_scale = layers.fully_connected(lhuc_output, int(net.shape[1]), activation_fn=tf.sigmoid,
                                                        scope=lhuc_scale_layer_scope)
                    net = net * lhuc_scale * 2.0
                if need_dropout:
                    net = tf.layers.dropout(net, rate=dropout_rate, noise_shape=None, seed=None,
                                            training=is_training, name=None)
            # record last layer tenor
            last_net = net
            with tf.variable_scope(name_or_scope="Logits") as deep_logits_scope:
                with arg_scope(base_ops.model_arg_scope(weight_decay=dnn_l2_reg)):
                    concat_layer = tf.concat([net], axis=1)
                    logits = layers.fully_connected(concat_layer, 1, activation_fn=None,
                                                    scope=deep_logits_scope)
    return net, logits
