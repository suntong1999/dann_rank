# encoding:utf-8
import tensorflow as tf
from tensorflow.contrib import layers


def subexpert_integration(input, name, l2_reg, subexpert_nums=5, subexpert_units='128,64'):
    """
    subexpert integration module
    """
    subexpert_units = list(map(int, subexpert_units.split(',')))
    subexperts = []
    for j in range(subexpert_nums):
        subexpert = input
        for i in range(len(subexpert_units)):
            subexpert = tf.layers.dense(inputs=subexpert, units=subexpert_units[i],
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        bias_initializer=tf.zeros_initializer(),
                                        name='subexpert_%s_%d_%d' % (name, j, i))
        subexperts.append(subexpert)
    subexperts = tf.concat([tf.expand_dims(se, axis=1) for se in subexperts], axis=1)  # None * 5 * 64
    gate_network = tf.contrib.layers.fully_connected(
        inputs=input,
        num_outputs=subexpert_nums,
        activation_fn=tf.nn.relu, \
        weights_regularizer=l2_reg)
    gate_network_shape = gate_network.get_shape().as_list()
    gate_network = tf.nn.softmax(gate_network, axis=1)
    gate_network = tf.reshape(gate_network, shape=[-1, gate_network_shape[1], 1])  # None * 5 * 1
    output = tf.multiply(subexperts, gate_network)  # None * 5 * 64
    output = tf.reduce_sum(output, axis=1)  # None * 64
    return output


def hinet_stack(input_features, scenario, src,  l2_reg=0.0,subexpert_units='128,192'):

    scenario_num = 3
    src_num = 11
    scenario_layer_lst = []
    for i in range(scenario_num):
        scenario_layer = subexpert_integration(input=input_features, name='scenario_layer_%d' % i,
                                               l2_reg=l2_reg, subexpert_nums=5,
                                               # subexpert_units='128,64')
                                               subexpert_units=subexpert_units)
        src_layer_lst = []
        for j in range(src_num):
            src_expert = subexpert_integration(input=scenario_layer, name='src_layer_%d_%d' % (i, j),
                                               l2_reg=l2_reg, subexpert_nums=5,
                                               # subexpert_units='128,64')
                                               subexpert_units=subexpert_units)

            src_layer_lst.append(tf.expand_dims(src_expert, axis=1))

        src_layer_lst = tf.concat(src_layer_lst, axis=1)  # None * 11 * 192
        print ('src_layer_lst', src_layer_lst)
        # get current scenario expert
        cur_src_index = tf.one_hot(indices=tf.reshape(tf.cast(src, dtype=tf.int64), [-1, ]) - 1,
                                   depth=src_num)  # None * 9

        cur_src_index = tf.expand_dims(cur_src_index, axis=2)  # None * 11 * 1
        print ('cur_src_index', cur_src_index)

        src_specific_expert = tf.multiply(src_layer_lst, cur_src_index)  # None * 11 * 192
        print ('src_specific_expert', src_specific_expert)

        src_specific_expert = tf.reduce_sum(src_specific_expert, axis=1, keepdims=True)  # None * 1 * 64
        src_specific_expert = tf.squeeze(src_specific_expert, axis=1)
        print ('src_specific_exper2', src_specific_expert)

        scenario_layer_lst.append(tf.expand_dims(src_specific_expert, axis=1))

    scenario_layer_lst = tf.concat(scenario_layer_lst, axis=1)  # None * 3 * 192
    print ('scenario_layer_lst', scenario_layer_lst)

    # get current scenario expert

    cur_scenario_index = tf.one_hot(indices=tf.reshape(tf.cast(scenario, dtype=tf.int64), [-1, ]) - 1,
                                    depth=scenario_num)  # None * 11

    cur_scenario_index = tf.expand_dims(cur_scenario_index, axis=-1)  # None * 3 * 1
    print ('cur_scenario_index', cur_scenario_index)

    scenario_specific_expert = tf.multiply(scenario_layer_lst, cur_scenario_index)  # None * 3 * 192
    print ('scenario_specific_expert', scenario_specific_expert)

    scenario_specific_expert = tf.reduce_sum(scenario_specific_expert, axis=1, keepdims=True)  # None * 1 * 192
    scenario_specific_expert = tf.squeeze(scenario_specific_expert, axis=1)

    print('scenario_specific_expert2', scenario_specific_expert)
    return scenario_specific_expert

def hinet_stack2(input_features, scenario, src,  l2_reg=0.0,subexpert_units='128,192'):

    scenario_num = 3
    src_num = 11
    scenario_layer_lst = []
    for i in range(scenario_num):
        scenario_layer = subexpert_integration(input=input_features, name='scenario_layer_%d' % i,
                                               l2_reg=l2_reg, subexpert_nums=5,
                                               # subexpert_units='128,64')
                                               subexpert_units=subexpert_units)
        src_layer_lst = []
        for j in range(src_num):
            src_expert = subexpert_integration(input=scenario_layer, name='src_layer_%d_%d' % (i, j),
                                               l2_reg=l2_reg, subexpert_nums=5,
                                               # subexpert_units='128,64')
                                               subexpert_units=subexpert_units)

            src_layer_lst.append(tf.expand_dims(src_expert, axis=1))

        src_layer_lst = tf.concat(src_layer_lst, axis=1)  # None * 11 * 192
        print ('src_layer_lst', src_layer_lst)
        # get current scenario expert
        cur_src_index = tf.one_hot(indices=tf.reshape(tf.cast(src, dtype=tf.int64), [-1, ]),
                                   depth=src_num)  # None * 9
        cur_src_index = tf.expand_dims(cur_src_index, axis=2)  # None * 11 * 1
        print ('cur_src_index', cur_src_index)
        src_specific_expert = tf.multiply(src_layer_lst, cur_src_index)  # None * 11 * 192
        print ('src_specific_expert', src_specific_expert)

        src_specific_expert = tf.reduce_sum(src_specific_expert, axis=1, keepdims=True)  # None * 1 * 64
        src_specific_expert = tf.squeeze(src_specific_expert, axis=1)
        #
        print ('src_specific_exper2', src_specific_expert)

        scenario_layer_lst.append(tf.expand_dims(src_specific_expert, axis=1))

    scenario_layer_lst = tf.concat(scenario_layer_lst, axis=1)  # None * 3 * 192
    print ('scenario_layer_lst', scenario_layer_lst)

    # get current scenario expert
    cur_scenario_index = tf.one_hot(indices=tf.reshape(tf.cast(scenario, dtype=tf.int64), [-1, ]),
                                    depth=scenario_num)  # None * 11
    cur_scenario_index = tf.expand_dims(cur_scenario_index, axis=-1)  # None * 3 * 1
    print ('cur_scenario_index', cur_scenario_index)
    scenario_specific_expert = tf.multiply(scenario_layer_lst, cur_scenario_index)  # None * 3 * 192
    print ('scenario_specific_expert', scenario_specific_expert)

    scenario_specific_expert = tf.reduce_sum(scenario_specific_expert, axis=1, keepdims=True)  # None * 1 * 192
    scenario_specific_expert = tf.squeeze(scenario_specific_expert, axis=1)

    print('scenario_specific_expert2', scenario_specific_expert)
    return scenario_specific_expert
# HINET
def hinet_hierarchy3(input_features, scenario, src,  l2_reg=0.0,subexpert_units='128,192'):
    """加入cond版本"""
    l2_reg = tf.contrib.layers.l2_regularizer(l2_reg)
    scenario_specific_expert = tf.cond(
            tf.logical_or(tf.equal(tf.size(scenario), 0), tf.equal(tf.size(src), 0)),
        lambda: layers.fully_connected(input_features, 192, activation_fn=tf.nn.tanh, weights_regularizer=l2_reg),
        lambda: hinet_stack(input_features, scenario, src,  l2_reg=l2_reg,subexpert_units=subexpert_units))
    print('scenario_specific_expert', scenario_specific_expert)
    return  scenario_specific_expert
def hinet_hierarchy4(input_features, scenario, src,  l2_reg=0.0,subexpert_units='128,192'):
    """加入cond版本"""
    l2_reg = tf.contrib.layers.l2_regularizer(l2_reg)
    # scenario_specific_expert = hinet_stack2(input_features, scenario, src,  l2_reg=l2_reg,subexpert_units=subexpert_units)
    scenario_specific_expert = tf.cond(
            tf.logical_or(tf.equal(tf.size(scenario), 0), tf.equal(tf.size(src), 0)),
        lambda: layers.fully_connected(input_features, 192, activation_fn=tf.nn.tanh, weights_regularizer=l2_reg),
        lambda: hinet_stack2(input_features, scenario, src,  l2_reg=l2_reg,subexpert_units=subexpert_units))
    print('scenario_specific_expert', scenario_specific_expert)
    return  scenario_specific_expert
def hinet_hierarchy5(input_features, scenario, src,  l2_reg=0.0,subexpert_units='128,192'):
    """加入cond版本"""
    l2_reg = tf.contrib.layers.l2_regularizer(l2_reg)
    scenario_specific_expert = hinet_stack2(input_features, scenario, src,  l2_reg=l2_reg,subexpert_units=subexpert_units)

    print('scenario_specific_expert', scenario_specific_expert)
    return  scenario_specific_expert
if __name__ == '__main__':
    input_features = tf.random_normal(shape=[2, 3])
    # int
    scenario = tf.random_uniform(shape=[2, ], minval=0, maxval=2, dtype=tf.int32)
    src = tf.random_uniform(shape=[2, ], minval=0, maxval=10, dtype=tf.int32)
    with tf.variable_scope('hinet', reuse=tf.AUTO_REUSE):
        output = hinet_hierarchy4(input_features, scenario, src, l2_reg=0.0,subexpert_units='3')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(output)
        print(sess.run(output))
