#encoding:utf-8
import tensorflow as tf
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


# HINET
def hinet(input_features, scenario_indicator,scenario_indicator_embedding,  l2_reg=0.0, scenario_num=12):
    l2_reg = tf.contrib.layers.l2_regularizer(l2_reg)
    with tf.variable_scope('shared-expert-part',reuse=tf.AUTO_REUSE):# 这个地方的scope层级错了，不过不影响训练，就这样吧
        scenario_shared_expert = subexpert_integration(input=input_features, name='scenario_shared_expert',
                                                       l2_reg=l2_reg, subexpert_nums=5,
                                                       subexpert_units='128,64')

        # scenario extract module
    scenario_experts = []
    with tf.variable_scope('scenario-extract-module-part',reuse=tf.AUTO_REUSE):
        # init scenario expert
        scenario_num = scenario_num
        scenario_specific_subexpert_units = '128,64'
        for j in range(scenario_num):
            scenario_expert = subexpert_integration(input=input_features, name='scenario_specific_expert_%d' % j,
                                                    l2_reg=l2_reg, subexpert_nums=5,
                                                    subexpert_units=scenario_specific_subexpert_units)
            scenario_experts.append(tf.expand_dims(scenario_expert, axis=1))  # None * 1 * 64
        scenario_experts = tf.concat(scenario_experts, axis=1)  # None * 9 * 64
        # get current scenario expert
        cur_scenario_index = tf.one_hot(indices=tf.reshape(tf.cast(scenario_indicator, dtype=tf.int64), [-1, ]) - 1,
                                        depth=scenario_num)  # None * 9
        cur_scenario_index = tf.expand_dims(cur_scenario_index, axis=2)  # None * 9 * 1
        scenario_specific_expert = tf.multiply(scenario_experts, cur_scenario_index)  # None * 9 * 64
        scenario_specific_expert = tf.reduce_sum(scenario_specific_expert, axis=1, keepdims=True)  # None * 1 * 64

        scenario_expert_gate = scenario_indicator_embedding

        scenario_gate_units = [128,64]
        scenario_gate_units.append(scenario_num)
        for i in range(len(scenario_gate_units)):
            scenario_expert_gate = tf.layers.dense(inputs=scenario_expert_gate, units=scenario_gate_units[i],
                                                   # activation=tf.nn.relu,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                   bias_initializer=tf.zeros_initializer(),
                                                   name='scenario_expert_gate_%d' % i)  # None * 9

        all_scenario_index = tf.ones_like(cur_scenario_index, dtype=tf.int64)
        all_scenario_index = all_scenario_index - \
                             tf.expand_dims(tf.one_hot(
                                 indices=tf.reshape(tf.cast(scenario_indicator, dtype=tf.int64), [-1, ]) - 1,
                                 depth=scenario_num, dtype=tf.int64), axis=2)
        scenario_expert_gate = tf.expand_dims(scenario_expert_gate, axis=2)
        scenario_expert_gate = tf.nn.softmax(scenario_expert_gate, axis=1)  # None * 9 * 1
        scenario_transfer_expert = tf.multiply(scenario_expert_gate, scenario_experts)  # None * 9 * 64
        scenario_transfer_expert = tf.multiply(scenario_transfer_expert,
                                               tf.cast(all_scenario_index, dtype=tf.float32))
        scenario_transfer_expert = tf.reduce_sum(scenario_transfer_expert, axis=1)  # None * 64

    # concat scenario-specific expert, scenario-aware expert and scenario-shared expert
    scenario_specific_expert = tf.squeeze(scenario_specific_expert, axis=1)
    scenario_out_concat = tf.concat([scenario_transfer_expert, scenario_specific_expert, scenario_shared_expert],
                                    axis=-1)
    return scenario_out_concat
if __name__ == '__main__':
    input_features = tf.random_normal(shape=[16, 3])
    scenario_indicator = tf.random_uniform(shape=[16, 1], minval=1, maxval=13, dtype=tf.int32)
    # scenario_indicator_embedding = tf.zeros(shape=[16, 4])
    scenario_indicator_embedding = tf.random_normal(shape=[16, 4])
    with tf.variable_scope('hinet', reuse=tf.AUTO_REUSE):
        output =hinet(input_features, scenario_indicator,scenario_indicator_embedding,  l2_reg=0.0, scenario_num=12)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(output))
        # 打印scenario_indicator_embedding的梯度
        print(sess.run(tf.gradients(output, scenario_indicator_embedding)))
