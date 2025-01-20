import logging as logger
import tensorflow as tf
from tensorflow.python.ops import array_ops

# from tensorflow.python.platform.tf_logging import _handler
import tensorflow.contrib.layers as layers
import tensorflow.contrib.opt as opt
from model_ops.tflog import tflogger as logging
from tensorflow.contrib.layers.python.layers import initializers

def reset_variables(collection_key=tf.GraphKeys.LOCAL_VARIABLES, matchname="auc/"):
    localv = tf.get_collection(collection_key)
    localv = [x for x in localv if matchname in x.name]
    retvops = [
        tf.assign(x, array_ops.zeros(shape=x.get_shape(), dtype=x.dtype))
        for x in localv
    ]
    if len(retvops) == 0:
        return None, None
    retvops = tf.tuple(retvops)
    return retvops, localv


def getActivationFunctionOp(act_name="relu"):
    if type(act_name) != str and type(act_name) != unicode:
        logging.warn("type(act_name) != str")
        logging.warn(type(act_name))
        return act_name

    if act_name.lower() == "relu":
        return tf.nn.relu
    elif act_name.lower() == "tanh":
        return tf.nn.tanh
    elif act_name.lower() == "lrelu":
        return lambda x: tf.nn.leaky_relu(x, alpha=0.01)
    elif act_name.lower() == "llrelu":
        return lambda x: tf.nn.leaky_relu(x, alpha=0.1)
    elif act_name.lower() == "prelu":
        return lambda x: parametric_relu(x)
    else:
        return tf.nn.relu


def parametric_relu(_x):
    alphas = tf.get_variable(
        "alpha",
        _x.get_shape()[-1],
        initializer=tf.constant_initializer(0.0),
        dtype=tf.float32,
    )
    from tensorflow.contrib.layers.python.layers.layers import (
        _add_variable_to_collections,
    )
    from tensorflow.python.framework import ops

    _add_variable_to_collections(alphas, [ops.GraphKeys.MODEL_VARIABLES], None)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg


def getInitOp(init_para, act_name="zero"):
    if type(act_name) != str and type(act_name) != unicode:
        logging.warn("type(act_name) != str")
        logging.warn(type(act_name))
        return act_name

    if act_name.lower() == "zero":
        return tf.zeros_initializer
    elif act_name.lower() == "constant":
        return tf.constant_initializer(init_para)
    elif act_name.lower() == "xavier":
        return initializers.xavier_initializer()
    else:
        return tf.zeros_initializer


def getOptimizer(
    training_config, global_step=None, learning_rate=None, learning_rate_decay_fn=None
):
    if training_config.optimizer == "AdagradDecay":
        if learning_rate == None:
            learning_rate = 0.01
        if learning_rate is not None and learning_rate_decay_fn is not None:
            if global_step is None:
                raise ValueError("global_step is required for learning_rate_decay_fn.")
            learning_rate = learning_rate_decay_fn(learning_rate, global_step)
        return tf.train.AdagradDecayOptimizer(
            learning_rate,
            global_step,
            accumulator_decay_step=training_config.decay_step,
            accumulator_decay_rate=training_config.decay_rate,
        )
    if training_config.optimizer == "AdamAsync":
        if learning_rate == None:
            learning_rate = 0.01
        if learning_rate is not None and learning_rate_decay_fn is not None:
            if global_step is None:
                raise ValueError("global_step is required for learning_rate_decay_fn.")
            learning_rate = learning_rate_decay_fn(learning_rate, global_step)
        return tf.train.AdamAsyncOptimizer(
            learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8
        )
    if training_config.optimizer.lower() == "ftrl":
        opti = lambda lr: tf.train.FtrlOptimizer(
            learning_rate=lr,
            initial_accumulator_value=0.1,  # more less , more sparse
            l1_regularization_strength=0.1,  # more large, more sparse
            l2_regularization_strength=0,
            use_locking=training_config.optimizer_use_lock,
        )
        return opti
    elif training_config.optimizer.lower() == "lazyadam":
        opti = lambda lr: opt.LazyAdamOptimizer(
            lr, use_locking=training_config.optimizer_use_lock
        )
        return opti
    elif training_config.optimizer.lower() == "momentum":
        opti = lambda lr: tf.train.MomentumOptimizer(
            lr,
            momentum=training_config.momentum,
            use_locking=training_config.optimizer_use_lock,
        )
        return opti
    elif training_config.optimizer.lower() == "AdaDelta":
        opti = lambda lr: tf.train.AdadeltaOptimizer(
            lr,
            training_config.adadelta_rho,
            training_config.adadelta_epsilon,
            use_locking=training_config.optimizer_use_lock,
        )
        return opti
    elif training_config.optimizer in layers.OPTIMIZER_CLS_NAMES:
        opti = lambda lr: layers.OPTIMIZER_CLS_NAMES[training_config.optimizer](
            lr, use_locking=training_config.optimizer_use_lock
        )
        return opti
    else:
        logging.warn("Optimizer [%s] not implemented" % training_config.optimizer)
        return training_config.optimizer


def getOptimizerVec(
    training_config,
    loss,
    global_step,
    learning_rate,
    model,
    learning_rate_decay_fn=None,
    increment_global_step=False,
    clip_gradients=None,
):
    if training_config.optimizer in layers.OPTIMIZER_CLS_NAMES:
        return [
            layers.optimize_loss(
                loss,
                global_step,
                learning_rate,
                optimizer=training_config.optimizer,
                learning_rate_decay_fn=learning_rate_decay_fn,
                variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
                increment_global_step=increment_global_step,
                clip_gradients=clip_gradients,
            )
        ]
    elif "LY" == training_config.optimizer:
        retv = []
        # dense layer with rmsprop
        retv.append(
            layers.optimize_loss(
                loss,
                global_step,
                learning_rate / 10000,
                optimizer="RMSProp",
                learning_rate_decay_fn=learning_rate_decay_fn,
                variables=tf.get_collection(model.collections_dnn_hidden_layer),
                increment_global_step=increment_global_step,
                clip_gradients=clip_gradients,
            )
        )
        # sparse layer with adagrad
        retv.append(
            layers.optimize_loss(
                loss,
                global_step,
                learning_rate,
                optimizer="Adagrad",
                learning_rate_decay_fn=learning_rate_decay_fn,
                variables=list(
                    set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
                    - set(tf.get_collection(model.collections_dnn_hidden_layer))
                ),
                increment_global_step=increment_global_step,
                clip_gradients=clip_gradients,
            )
        )
        return retv
    else:
        logging.warn("getOptimizerVec not impl [%s]" % str(training_config.optimizer))
        return [
            layers.optimize_loss(
                loss,
                global_step,
                learning_rate,
                optimizer=training_config.optimizer,
                learning_rate_decay_fn=learning_rate_decay_fn,
                variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
                increment_global_step=increment_global_step,
                clip_gradients=clip_gradients,
            )
        ]


def add_auc(
    labels,
    predictions,
    name,
    weights=None,
    runner=None,
    summary_dict=None,
    summary_name=None,
    add_to_log=False,
):
    print("calculate auc:")
    print("labels:",labels)
    print("predictions:",predictions)
    current_auc, auc_update_op = tf.metrics.auc(
        labels=labels,
        predictions=predictions,
        weights=weights,
        num_thresholds=2000,
        name=name,
    )
    if runner is not None:
        runner.add_train_ops([auc_update_op])
        runner.add_evaluate_ops([auc_update_op])
        if add_to_log:
            runner.add_log_ops([summary_name], [current_auc])
    if summary_dict is not None:
        summary_dict[summary_name] = current_auc


