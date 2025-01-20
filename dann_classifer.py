from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
import tensorflow.contrib.layers as layers


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls
        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            if grad is None:
                raise ValueError("Gradient is None")
            return [tf.negative(grad) * l]
        
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
            
        self.num_calls += 1
        return y
    



class DANNModel(object):
    def __init__(self):
        # self.alpha = alpha
        self.class_classifier = None
        self.domain_classifier = None
        self.l = tf.placeholder(tf.float32, [])
    
    def build_class_classifier(self, input_feature):
        with tf.variable_scope('class_classifier', reuse=tf.AUTO_REUSE):
            c_fc1_weights = tf.get_variable('c_fc1_weights', shape=[192, 100],
                                            initializer=tf.contrib.layers.xavier_initializer())
            c_fc1_biases = tf.get_variable('c_fc1_biases', shape=[100], initializer=tf.zeros_initializer())

            def classifier_logic(x):
                print("x shape before reshape: ", x.shape)
                # reshaped_x = tf.reshape(x, [-1, 50 * 4 * 4])
                reshaped_x = tf.reshape(x, [-1, 192])
                print("x shape after reshape: ", reshaped_x.shape)
                c_fc1_output = tf.nn.relu(tf.matmul(reshaped_x, c_fc1_weights) + c_fc1_biases)
                # c_fc1_output = tf.nn.relu(tf.matmul(tf.reshape(x, [-1, 50 * 4 * 4]), c_fc1_weights) + c_fc1_biases)
                c_bn1_output = layers.batch_norm(c_fc1_output, is_training=True, scope='c_bn1')

                c_drop1_output = tf.nn.dropout(c_bn1_output, keep_prob=0.5)

                c_fc2_weights = tf.get_variable('c_fc2_weights', shape=[100, 100],
                                                initializer=tf.contrib.layers.xavier_initializer())
                c_fc2_biases = tf.get_variable('c_fc2_biases', shape=[100], initializer=tf.zeros_initializer())
                c_fc2_output = tf.nn.relu(tf.matmul(c_drop1_output, c_fc2_weights) + c_fc2_biases)

                c_bn2_output = layers.batch_norm(c_fc2_output, is_training=True, scope='c_bn2')

                c_fc3_weights = tf.get_variable('c_fc3_weights', shape=[100, 2],
                                                initializer=tf.contrib.layers.xavier_initializer())
                c_fc3_biases = tf.get_variable('c_fc3_biases', shape=[2], initializer=tf.zeros_initializer())
                c_fc3_output = tf.nn.log_softmax(tf.matmul(c_bn2_output, c_fc3_weights) + c_fc3_biases)

                return c_fc3_output

            return classifier_logic
        
    def build_domain_classifier(self, input_feature):
        with tf.variable_scope('domain_classifier', reuse=tf.AUTO_REUSE):
            d_fc1_weights = tf.get_variable('d_fc1_weights', shape=[192, 100],
                                            initializer=tf.contrib.layers.xavier_initializer())
            d_fc1_biases = tf.get_variable('d_fc1_biases', shape=[100], initializer=tf.zeros_initializer())
            
            def classifier_logic1(x):
                flip_gradient = FlipGradientBuilder()
                feat_x = flip_gradient(x, self.l)
                # d_fc1_output = tf.nn.relu(tf.matmul(tf.reshape(x, [-1, 50 * 4 * 4]), d_fc1_weights) + d_fc1_biases)
                d_fc1_output = tf.nn.relu(tf.matmul(tf.reshape(feat_x, [-1, 192]), d_fc1_weights) + d_fc1_biases)
                d_bn1_output = layers.batch_norm(d_fc1_output, is_training=True, scope='d_bn1')

                d_fc2_weights = tf.get_variable('d_fc2_weights', shape=[100, 2],
                                                initializer=tf.contrib.layers.xavier_initializer())
                d_fc2_biases = tf.get_variable('d_fc2_biases', shape=[2], initializer=tf.zeros_initializer())
                d_fc2_output = tf.nn.log_softmax(tf.matmul(d_bn1_output, d_fc2_weights) + d_fc2_biases)

                return d_fc2_output
            return classifier_logic1

    def forward(self, feature):
        assert callable(self.class_classifier), "self.class_classifier is not a callable object"
        assert callable(self.domain_classifier), "self.domain_classifier is not a callable object"
        print("feature:",feature)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(feature)
        return class_output, domain_output

    def build_model(self, input_feature):
        self.class_classifier = self.build_class_classifier(input_feature)
        self.domain_classifier = self.build_domain_classifier(input_feature)
        print("Type of self.class_classifier:", type(self.class_classifier))
        print("Type of self.domain_classifier:", type(self.domain_classifier))

        

