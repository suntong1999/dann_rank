# encoding: utf-8
from __future__ import print_function
from tensorflow.contrib import layers
from model_ops import ops as base_ops
from layers.attention import feedforward_v1
from layers.mmoe import mtl_ple_v1
from utils.fg_mid import fg_mid, get_feature, get_seq_feature, concat_seq_features
from layers.tower import tower2
from collections import OrderedDict
import tensorflow as tf
from model_ops.utils import add_auc
from layers.attention import multihead_self_attention, target_attention
from layers.dann_classifier import DANNModel
import random
import numpy as np

version = tf.VERSION.split(".")
# 1.4.0,1.12.0,2.0.0
if not (int(version[0]) == 1 and int(version[1]) <= 8):
    import pai


def set_shape_optimize():
    # https://yuque.antfin-inc.com/rtp/wtm2oh/vsfvgm
    tf.get_default_graph().set_shape_optimize(False)


class DNNModel(object):
    def __init__(self, global_conf, FLAGS):

        # model config
        self.FLAGS = FLAGS
        if self.FLAGS.mode_run == "local":
            from utils.runner import LocalRunner

            self.runner = LocalRunner(FLAGS)
        else:
            from utils.runner import Runner

            self.runner = Runner(FLAGS)
        self.global_conf = global_conf
        self.config = global_conf["model"]

        # job config
        self.ps_num = len(self.FLAGS.ps_hosts.split(","))
        self.max_checkpoints_to_keep = self.FLAGS.max_checkpoints_to_keep

        # hyper-parameters config
        self.dnn_l2_reg = self.config["dnn_l2_reg"]
        self.need_dropout = self.config["need_dropout"]
        self.dropout_rate = self.config["dropout_rate"]
        self.learning_rate = self.config["learning_rate"]
        self.max_grad_norm = self.config["max_grad_norm"]
        self.ortho_init_scale = self.config["ortho_init_scale"]
        self.dnn_hidden_units = self.config["dnn_hidden_units"]
        self.dnn_hidden_units_act_op = self.config["dnn_hidden_units_act_op"]
        self.embedding_partition_size = self.config["embedding_partition_size"]
        
        

        (
            self.fg,
            self.block_column_dict,
            self.seq_block_column_dict,
            self.feature_column_dict,
        ) = fg_mid(global_conf, global_conf["fg_path"])


        # A string Tensor with shape [batch_size, 1].
        self.id = None

        # A batch of dict of feature-name & feature-value.
        self.features = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.loss = None

        # Define model variables collection，应该没什么用
        self.collections_dnn_hidden_layer = "collections_dnn_hidden_layer"
        self.collections_dnn_hidden_output = "collections_dnn_hidden_output"

        # model name
        self.name = "dann_pvpay"

    def build_placeholder(self):
        try:
            training = tf.get_default_graph().get_tensor_by_name("training:0")
        except KeyError as e:
            print(e)
            training = tf.placeholder(tf.bool, name="training")
        self.is_training = training

    def build_inputs(self, batch_data, use_fg=True, local_mode=False):
        # use fg features to embedding
        with tf.name_scope("%s_Input_Pipeline" % self.name):
            if local_mode:
                self.build_local_inputs(batch_data, use_fg)
            else:
                self.build_training_inputs(batch_data)

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES],
        )

        self.global_step = global_step
        self.global_step_reset = tf.assign(self.global_step, 0)

    def process_image_feature(self, pid_embed_input_layer, query_embed_input_layer):
        with tf.variable_scope("pid_query_cross_layer"):
            pid_embed_input_layer = feedforward_v1(
                pid_embed_input_layer,
                num_units=[128, 16],
                activation_fn=tf.nn.tanh,
                variables_collections=None,
                outputs_collections=None,
                is_training=self.get_is_training(),
            )

            query_embed_input_layer = feedforward_v1(
                query_embed_input_layer,
                num_units=[128, 16],
                activation_fn=tf.nn.tanh,
                variables_collections=None,
                outputs_collections=None,
                is_training=self.get_is_training(),
            )

            pq_cross_layer = layers.fully_connected(
                tf.concat([query_embed_input_layer, pid_embed_input_layer], axis=1),
                16,
                normalizer_fn=layers.batch_norm,
                normalizer_params={
                    "scale": True,
                    "is_training": self.get_is_training(),
                },
            )

            return pid_embed_input_layer, query_embed_input_layer, pq_cross_layer

    def get_tt_attention_score(self, trigger_sparse_layer, target_sparse_layer):
        trigger_sparse_layer = layers.fully_connected(
            trigger_sparse_layer, 60, activation_fn=None
        )
        target_sparse_layer = layers.fully_connected(
            target_sparse_layer, 60, activation_fn=None
        )

        trigger_sparse_layer = tf.reshape(trigger_sparse_layer, [-1, 5, 12])
        target_sparse_layer = tf.reshape(target_sparse_layer, [-1, 5, 12])

        attention_score = tf.reduce_sum(
            trigger_sparse_layer * target_sparse_layer, axis=-1
        )

        attention_score = layers.batch_norm(
            attention_score, scale=True, is_training=self.get_is_training()
        )

        return attention_score

    def build_model(self):
        if not (self.features and isinstance(self.features, dict)):
            raise ValueError("features must be defined and must be a dict.")
        if not self.dnn_hidden_units:
            raise ValueError("configuration dnn_hidden_units must be defined.")

        with tf.variable_scope(
            name_or_scope="input_from_feature_columns",
            partitioner=base_ops.partitioner(
                ps_num=self.ps_num, mem=self.embedding_partition_size
            ),
            reuse=tf.AUTO_REUSE,
        ) as scope:
            self.sku_sparse_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "sku_sparse",
                BN=False,
                # Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.sku_dense_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "sku_dense",
                BN=False,
                # Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.item_sparse_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "item_sparse",
                BN=False,
                # Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.item_dense_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "item_dense",
                BN=False,
                # Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.query_embed_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "query_embed",
                BN=False,
                Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.pid_embed_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "pid_embed",
                BN=False,
                Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.context_dense_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "context_dense",
                # Print=True,
                BN=False,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.user_sparse_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "user_sparse",
                BN=False,
                # Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.domain_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "domain",
                BN=False,
                Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.domain_indicator_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "domain_indicator",
                Print=True,
                BN=False,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.search_scenario_indicator_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "search_scenario_indicator",
                BN=False,
                Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.src_indicator_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "src_indicator",
                BN=False,
                Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
        with tf.variable_scope(
            name_or_scope="input_from_feature_columns",
            partitioner=base_ops.partitioner(
                ps_num=self.ps_num, mem=self.embedding_partition_size
            ),
            reuse=tf.AUTO_REUSE,
        ) as scope:
            trigger_sku_price_layer = layers.input_from_feature_columns(
                self.features,
                [self.feature_column_dict["trigger_sku_current_price"]],
                scope=scope,
            )
            sku_price_layer = layers.input_from_feature_columns(
                self.features,
                [self.feature_column_dict["sku_current_price"]],
                scope=scope,
            )
#######################################################################

            self.sku_sparse_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "sku_sparse",
                BN=False,
                # Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.sku_dense_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "sku_dense",
                BN=False,
                # Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.item_sparse_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "item_sparse",
                BN=False,
                # Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.item_dense_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "item_dense",
                BN=False,
                # Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.query_embed_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "query_embed",
                BN=False,
                Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.pid_embed_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "pid_embed",
                BN=False,
                Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.context_dense_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "context_dense",
                # Print=True,
                BN=False,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.user_sparse_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "user_sparse",
                BN=False,
                # Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.domain_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "domain",
                BN=False,
                Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.domain_indicator_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "domain_indicator",
                Print=True,
                BN=False,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.search_scenario_indicator_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "search_scenario_indicator",
                BN=False,
                Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
            self.src_indicator_input_layer = get_feature(
                self.features,
                self.block_column_dict,
                "src_indicator",
                BN=False,
                Print=True,
                is_training=self.get_is_training(),
                scope=scope,
            )
        with tf.variable_scope(
            name_or_scope="input_from_feature_columns",
            partitioner=base_ops.partitioner(
                ps_num=self.ps_num, mem=self.embedding_partition_size
            ),
            reuse=tf.AUTO_REUSE,
        ) as scope:
            trigger_sku_price_layer = layers.input_from_feature_columns(
                self.features,
                [self.feature_column_dict["trigger_sku_current_price"]],
                scope=scope,
            )
            sku_price_layer = layers.input_from_feature_columns(
                self.features,
                [self.feature_column_dict["sku_current_price"]],
                scope=scope,
            )


########################################################################################
        with tf.variable_scope(
            name_or_scope="MainNet",
            partitioner=base_ops.partitioner(
                ps_num=self.ps_num, mem=self.embedding_partition_size
            ),
            reuse=tf.AUTO_REUSE,
        ) as scope:
            # dequantize
            from utils.quantize import dequantize

            self.pid_embed_input_layer = dequantize(self.pid_embed_input_layer)
            # trigger and sku price diff layer
            self.price_diff_layer = tf.divide(
                sku_price_layer - trigger_sku_price_layer,
                tf.abs(trigger_sku_price_layer) + 1e-12,
            )
            with tf.variable_scope(name_or_scope="BN", reuse=tf.AUTO_REUSE) as scope:
                self.price_diff_layer = layers.batch_norm(
                    self.price_diff_layer,
                    scale=True,
                    is_training=self.get_is_training(),
                )
                # bn for item_dense_input_layer
                self.sku_sparse_input_layer = layers.batch_norm(
                    self.sku_sparse_input_layer,
                    scale=True,
                    is_training=self.get_is_training(),
                )
                self.sku_dense_input_layer = layers.batch_norm(
                    self.sku_dense_input_layer,
                    scale=True,
                    is_training=self.get_is_training(),
                )
                self.item_sparse_input_layer = layers.batch_norm(
                    self.item_sparse_input_layer,
                    scale=True,
                    is_training=self.get_is_training(),
                )
                self.item_dense_input_layer = layers.batch_norm(
                    self.item_dense_input_layer,
                    scale=True,
                    is_training=self.get_is_training(),
                )
                self.query_embed_input_layer = layers.batch_norm(
                    self.query_embed_input_layer,
                    scale=True,
                    is_training=self.get_is_training(),
                )
                self.pid_embed_input_layer = layers.batch_norm(
                    self.pid_embed_input_layer,
                    scale=True,
                    is_training=self.get_is_training(),
                )
                self.context_dense_input_layer = layers.batch_norm(
                    self.context_dense_input_layer,
                    scale=True,
                    is_training=self.get_is_training(),
                )

                self.user_sparse_input_layer = layers.batch_norm(
                    self.user_sparse_input_layer,
                    scale=True,
                    is_training=self.get_is_training(),
                )
                self.domain_input_layer = layers.batch_norm(
                    self.domain_input_layer,
                    scale=True,
                    is_training=self.get_is_training(),
                )

            # process pid embedding and query emebdding
            (
                self.pid_embed_input_layer,
                self.query_embed_input_layer,
                self.pq_cross_layer,
            ) = self.process_image_feature(
                self.pid_embed_input_layer, self.query_embed_input_layer
            )

            self.trigger_target_attention_score = self.get_tt_attention_score(
                self.item_sparse_input_layer, self.context_dense_input_layer
            )
            with tf.variable_scope(
                name_or_scope="%s-Score-Network" % self.name,
                partitioner=base_ops.partitioner(
                    ps_num=self.ps_num, mem=self.embedding_partition_size
                ),
            ):
                input = tf.concat(
                    [
                        self.sku_sparse_input_layer,
                        self.sku_dense_input_layer,
                        self.item_sparse_input_layer,
                        self.item_dense_input_layer,
                        self.context_dense_input_layer,
                        self.pid_embed_input_layer,
                        self.query_embed_input_layer,
                        self.pq_cross_layer,
                        self.price_diff_layer,
                        self.trigger_target_attention_score,
                        self.domain_input_layer,
                    ],
                    axis=1,
                )

                tf.add_to_collection(self.collections_dnn_hidden_output, input)
                from layers.hinet import hinet

                # INPUT: bs,input_dim

                input = hinet(
                    input,
                    self.domain_indicator_input_layer,
                    self.domain_input_layer,
                    self.dnn_l2_reg,
                )

                # hinet_output : bs,output_dim
                from layers.hinet_hsa_xg import hinet_hierarchy4

                # 转为int32
                self.search_scenario_indicator_input_layer = tf.cast(
                    self.search_scenario_indicator_input_layer, tf.int32
                )
                self.src_indicator_input_layer = tf.cast(
                    self.src_indicator_input_layer, tf.int32
                )
                input2 = hinet_hierarchy4(
                    input,
                    self.search_scenario_indicator_input_layer,
                    self.src_indicator_input_layer,
                    self.dnn_l2_reg,
                )

                self.feature = tf.maximum(input, input2)

                
                # self.model = DANNModel(alpha=self.alpha)
                self.head_model = DANNModel()

                self.head_model.build_model(self.feature)

                self.class_output, self.domain_output = self.head_model.forward(self.feature)
                
               
                


    def predict_op(self):
        with tf.name_scope("Predict"):
            self.pay_pred = self.class_output[:, :1]
            self.pay_pred = tf.sigmoid(self.pay_pred)
            
            self.pay_pred_printed = tf.Print(self.pay_pred, [self.pay_pred], message='pred Output:')
    
            print("self.pay_pred:",self.pay_pred)


    def cpp_rank_loss(self, logits, labels):
        pairwise_logits = logits - tf.transpose(logits)
        pairwise_mask = tf.greater(labels - tf.transpose(labels), 0)
        pairwise_logits = tf.boolean_mask(pairwise_logits, pairwise_mask)
        pairwise_psudo_labels = tf.ones_like(pairwise_logits)
        rank_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=pairwise_logits, labels=pairwise_psudo_labels
            )
        )
        # set rank loss to zero if a batch has no positive sample.
        rank_loss = tf.where(tf.is_nan(rank_loss), tf.zeros_like(rank_loss), rank_loss)
        return rank_loss

    def loss_op(self):
        with tf.name_scope("loss"):
            rounded_label = tf.cast(tf.round(self.all_domain_cvr_label_merge), tf.int32)
            self.class_label = tf.squeeze(rounded_label, axis=-1)
            batch_size = tf.shape(self.domain_label)[0]
            self.domain_label_post = tf.reshape(self.domain_label, [batch_size])
            self.domain_label_post = tf.cast(self.domain_label_post, tf.int32)
        
            loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.class_output, labels=self.class_label)
            loss_domain = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.domain_output, labels=self.domain_label_post)
            
            batch_size = tf.shape(self.domain_label)[0]
            
            domain_weight = self.domain_output[:, 0]
            self.normalized_domain_weight = tf.nn.softmax(domain_weight)
            
            # self.domain_weight = (tf.exp(self.domain_output[:, 0]) + 1e-6) / (1.0 + 1e-6)
            self.err_label = loss_class * self.normalized_domain_weight
            self.err_label = tf.reduce_mean(self.err_label)
            self.err_domain = loss_domain
            self.err_domain = tf.reduce_mean(self.err_domain)
            self.loss = self.err_label + self.err_domain*0.5




    def _add_histograms(self, gradients_clip, clip_gradients):
        from tensorflow.python.framework import ops
        from tensorflow.python.ops import clip_ops

        for gradient, variable in gradients_clip:
            if isinstance(gradient, ops.IndexedSlices):
                grad_values = gradient.values
            else:
                grad_values = gradient

            if grad_values is not None:
                var_name = variable.name.replace(":", "_")
                if "part_0_0" not in var_name:
                    continue
                tf.summary.histogram("gradients/%s" % var_name, grad_values)
                tf.summary.scalar(
                    "gradient_norm/%s" % var_name, clip_ops.global_norm([grad_values])
                )

        if clip_gradients is not None:
            tf.summary.scalar(
                "global_norm/clipped_gradient_norm",
                clip_ops.global_norm(list(zip(*gradients_clip))[0]),
            )

    def training_op(self):
        print("params={")
        params = tf.trainable_variables()
        # for p in params:
        #     print(p)
        print(len(params))
        print("}")

        grads = tf.gradients(self.loss, params)
        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        grads = list(zip(grads, params))

        # check grad_norm
        with tf.name_scope("Summary"):
            tf.summary.scalar(name="grad_norm", tensor=_grad_norm)
        self._add_histograms(grads, self.max_grad_norm)

        version = tf.VERSION.split(".")
        # 1.4.0,1.12.0,2.0.0
        if int(version[0]) == 1 and int(version[1]) <= 8:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        else:
            optimizer = tf.train.AdamAsyncOptimizer(learning_rate=self.learning_rate)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print("update_ops: %s" % len(self.update_ops))
        with tf.control_dependencies(self.update_ops):
            self.train_op = optimizer.apply_gradients(
                grads, global_step=self.global_step
            )

    def setup_saver(self):
        with tf.device(tf.train.replica_device_setter(ps_tasks=4)):
            saver = tf.train.Saver()

    def mark_output(self):
        with tf.name_scope("Mark_Output"):
            self._rank_predict = tf.identity(self.pay_pred * 1e5, name="rank_predict_pay")


    def summary(self, local_mode=False):
        # 构建summary
        summary_dict = OrderedDict()
        # 生成ops
        with tf.name_scope("Metrics"):
            if self.FLAGS.mode_run == "local":
                self.pvpay_current_auc, self.pvpay_auc_update_op = tf.metrics.auc(
                    labels=self.all_domain_cvr_label_merge,
                    predictions=self.pay_pred,
                    weights=self.is_image_search_label,
                    num_thresholds=2000,
                    name=self.name + "-auc",
                )
            else:
                worker_device = "/job:worker/task:{}".format(self.FLAGS.task_index)
                with tf.device(worker_device):
                    print("labels=self.all_domain_cvr_label_merge:", self.all_domain_cvr_label_merge.shape,self.all_domain_cvr_label_merge)
                    print("predictions=self.pay_pred:",self.pay_pred.shape,self.pay_pred)
                    print("weights=self.is_image_search_label:",self.is_image_search_label.shape,self.is_image_search_label)
                    
                    add_auc(
                        labels=self.all_domain_cvr_label_merge,
                        predictions=self.pay_pred,
                        name=self.name + "-pvpay_auc",
                        weights=self.is_image_search_label,
                        runner=self.runner,
                        summary_dict=summary_dict,
                        summary_name="pvpay_auc",
                        add_to_log=True,
                    ) 
                
        summary_dict["global_step"] = self.global_step
        summary_dict["label_loss"] = self.err_label
        summary_dict["domain_loss"] = self.err_domain
        summary_dict["loss"] = self.loss
        summary_dict["pay_pred_mean"] = tf.reduce_mean(self.pay_pred)

        with tf.name_scope("Summary"):
            for name, value in summary_dict.items():
                print("name:",name)
                print("value:",value)
                tf.summary.scalar(name="scalar/" + name, tensor=tf.reduce_mean(value))
                # tf.summary.scalar(name="scalar/" + name, tensor=value)
            for name, value in summary_dict.items():
                tf.summary.histogram(name="histogram/" + name, values=value)
        # 筛选部分ops打印log
        log_ops = [[], []]
        log_keys = [
            "label_loss",
            "domain_loss",
            "loss",
            "global_step",
            "pay_pred_mean",
        ]

        for k in log_keys:
            if k in summary_dict:
                log_ops[0].append(k)
                log_ops[1].append(summary_dict[k])
        self.runner.add_log_ops(log_ops[0], log_ops[1])

        # 再筛选部分ops到dataworks,最多五个
        dataworks_ops = [[], []]
        dataworks_keys = [
            "pvpay_auc",
            "loss", 
            "pay_pred_mean"
        ]
        for k in dataworks_keys:
            if k in summary_dict:
                dataworks_ops[0].append(k)
                dataworks_ops[1].append(summary_dict[k])
        self.runner.set_summary_op(dataworks_ops)

        # 添加到各自的ops集合中
        self.runner.add_train_ops(
            [self.train_op,]
        )
        self.runner.add_inference_ops(
            [
                self.id,
                self.all_domain_cvr_label_merge,
                self.domain_label,
                self.pay_pred,
            ]
        )
        merged_summary = tf.summary.merge_all()
        return merged_summary

    def trace_op(self):
        """
        trace feature input and feature output
        :return:
        """
        with tf.variable_scope(
            name_or_scope="input_from_feature_columns",
            partitioner=base_ops.partitioner(
                self.ps_num, mem=self.embedding_partition_size
            ),
            reuse=tf.AUTO_REUSE,
        ) as scope:
            columns = self.user_column + self.item_column + self.context_column
            for column in columns:
                feature_name = column.name
                feature_inputs = column.config.keys()
                # not support for two input
                assert len(feature_inputs) == 1
                feature_original = self.features[feature_inputs[0]]
                feature_output = layers.input_from_feature_columns(
                    self.features, [column], scope=scope
                )
                self.trace[feature_name] = [feature_original, feature_output]

    def build_local_inputs(self, batch_data, use_fg):
        if use_fg:
            self.build_training_inputs(batch_data)
            return
        features = {
            "user_sex": batch_data[0],
            "origin_keyword_hash": batch_data[1],
            "s_nid_ctr_30": tf.string_to_number(batch_data[3], out_type=tf.float32),
            "label": tf.reshape(
                tf.string_to_number(batch_data[4], out_type=tf.float32), [-1, 1]
            ),
            "click_weight": tf.reshape(
                tf.string_to_number(batch_data[5], out_type=tf.float32), [-1, 1]
            ),
        }
        self.features = features
        self.label = features["label"]

    def build_training_inputs(self, batch_data):
        if self.FLAGS.prefetch > 0:
            batch_data = pai.data.prefetch(
                batch_data,
                capacity=self.FLAGS.prefetch_capacity,
                num_threads=self.FLAGS.prefetch_num_threads,
            )
        self.id = tf.reshape(batch_data[0], [-1, 1])
        self.labels = tf.strings.split(batch_data[1], sep=";")
        self.labels = tf.sparse.to_dense(self.labels, default_value="0")
        import rtp_fg

        self.features = rtp_fg.parse_genreated_fg(
            self.fg.get_feature_conf, batch_data[2]
        )
        concat_seq_features(self.fg, self.global_conf, self.features)


        print("-------------\nfeatures keys len:", len(self.features.keys()))
        
        print("self.labels[:, 1]:",self.labels[:, 1].shape)

        self.ctr_label = tf.reshape(
            tf.string_to_number(self.labels[:, 0], out_type=tf.float32), [-1, 1]
        )
        self.cvr_mask = tf.to_float(tf.greater(self.ctr_label, 0.5))
        self.atc_label = tf.reshape(
            tf.string_to_number(self.labels[:, 1], out_type=tf.float32), [-1, 1]
        )
        self.cvr_label = tf.reshape(
            tf.string_to_number(self.labels[:, 2], out_type=tf.float32), [-1, 1]
        )
        self.all_domain_ctr_label = tf.reshape(
            tf.cast(self.features["all_domain_ctr_label"], tf.float32), [-1, 1]
        )

        self.all_domain_cvr_label = tf.reshape(
            tf.cast(self.features["all_domain_cvr_label"], tf.float32), [-1, 1]
        )
        # 如果在全域成交了，那么默认全域点击也是1，为了和原来的成交必有点击（cvr auc weight：ctr_label）逻辑一致
        self.all_domain_ctr_label = tf.maximum(
            self.all_domain_cvr_label, self.all_domain_ctr_label
        )
        self.all_domain_ctr_label_merge = tf.maximum(
            self.ctr_label, self.all_domain_ctr_label
        )
        # cvr_label和all_domain_cvr_label取或
        self.all_domain_cvr_label_merge = tf.maximum(
            self.cvr_label, self.all_domain_cvr_label
        )
        
        self.domain_label = tf.where(
            tf.logical_and(tf.equal(self.cvr_label, 0), tf.equal(self.all_domain_cvr_label_merge, 1)),
            tf.ones_like(self.cvr_label, dtype=tf.float32),  # 满足条件赋值为1
            tf.zeros_like(self.cvr_label, dtype=tf.float32)  # 不满足条件赋值为0
        )
#         1-global 0-local

        self.is_image_search_label = tf.reshape(
            tf.string_to_number(self.labels[:, 3], out_type=tf.float32), [-1, 1]
        )


    @property
    def model_name(self):
        return self.name

    def get_is_training(self):
        try:
            training = tf.get_default_graph().get_tensor_by_name("training:0")
        except KeyError as e:
            training = tf.placeholder(tf.bool, name="training")
        return training

    def build(self, batch_data, use_fg=True, local_mode=False):
        """Creates all ops for training and evaluation."""
        set_shape_optimize()
        self.build_placeholder()
        self.build_inputs(batch_data, use_fg, local_mode)
        self.setup_global_step()
        self.build_model()
        self.predict_op()
        self.loss_op()
        self.training_op()
        self.setup_saver()
        self.mark_output()
        self.summary(local_mode)  
