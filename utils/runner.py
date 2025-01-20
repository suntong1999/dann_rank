# encoding: utf-8
from abc import ABCMeta, abstractmethod
from mode import ModeKeys
from utils.util import logger, metric_table_logger
import tensorflow as tf
import numpy as np
import os
import sys
import time
from tabulate import tabulate
from tensorflow.python.client import timeline
from tensorflow.python.lib.io import file_io


class BaseRunner(object):
    __metaclass__ = ABCMeta

    def __init__(self, FLAGS, *args, **kwargs):
        self.FLAGS = FLAGS
        self._train_ops = []
        self._inference_ops = []
        self._evaluate_ops = []
        self._log_ops = [[], []]
        self._debug_ops = [[], []]

        self.model_start_flag = 0
        self.local_train_step = 0
        self.local_test_step = 0
        self.local_predict_step = 0
        self.local_last_step = 0
        self.during_step = 200 if self.FLAGS.mode_run != ModeKeys.LOCAL else 1
        self.odps_writer = None
        self.table_row_count = 0
        if self.FLAGS.timeline:
            self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()
        else:
            self.run_options = None
            self.run_metadata = None

    def add_train_ops(self, ops):
        self._train_ops.extend(ops)

    def add_inference_ops(self, ops):
        self._inference_ops = ops

    def add_evaluate_ops(self, ops):
        self._evaluate_ops.extend(ops)

    def add_log_ops(self, names, ops):
        assert len(names) == len(ops), "name number must match with ops"
        self._log_ops[0].extend(names)
        self._log_ops[1].extend(ops)

    def add_debug_ops(self, names, ops):
        assert len(names) == len(ops), "name number must match with ops"
        self._debug_ops[0].extend(names)
        self._debug_ops[1].extend(ops)

    def before_run(self, sess, *args, **kwargs):
        pass

    def table_progress(self, done, all):
        return format(float(all - done) / all * 100, "0.0f")

    def run(self, sess, model, table_row_count, *args, **kwargs):
        self.table_row_count = table_row_count
        if self.FLAGS.mode_run == ModeKeys.LOCAL:
            return self._train(sess,model)
        elif self.FLAGS.mode_run == ModeKeys.PREDICT:
            return self._inference(sess)
        elif self.FLAGS.mode_run == ModeKeys.TRAIN:
            if self.FLAGS.task_index == 0:  # worker 0 for validation
                return self._evaluate(sess)
            else:  # other workers for training
                return self._train(sess,model)
        elif self.FLAGS.mode_run == ModeKeys.EVAL:
            return self._evaluate(sess)
        else:
            raise Exception("Unsupported Mode:{}".format(self.FLAGS.mode_run))

    def after_run(self, sess, *args, **kwargs):
        if self.odps_writer is not None:
            self.odps_writer.close()

    def _train(self, sess, model, *args, **kwargs):
        
        print("Runner._train: self.local_train_step: ", self.local_train_step)
        print("Runner._train: tf.train.get_global_step: ", tf.train.get_global_step(), tf.train.get_global_step().type)
        
        p = float(self.local_train_step)/tf.train.get_global_step()
        l = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 0.01 / (1. + 10 * p)**0.75
        feed_dict = {"training:0": True, model.head_model.l: l} 
        
        result = sess.run(self._log_ops[1] + self._train_ops, feed_dict=feed_dict)
        if self.local_train_step % self.during_step == 0:
            logger(
                "Train:{}".format(
                    ", ".join(
                        [
                            "{}={:.7g}".format(k, v)
                            for k, v in zip(self._log_ops[0], result)
                        ]
                    )
                )
            )
            sys.stdout.flush()
        self.local_train_step += 1
        return False

    def _evaluate(self, sess, *args, **kwargs):
        feed_dict = {"training:0": False}
        result = sess.run(
            self._log_ops[1] + self._evaluate_ops + [tf.train.get_global_step()],
            feed_dict=feed_dict,
        )
        if self.local_test_step % self.during_step == 0:
            logger(
                "Validation:{}".format(
                    ", ".join(
                        [
                            "{}={:.7g}".format(k, v)
                            for k, v in zip(self._log_ops[0], result)
                        ]
                    )
                )
            )
            sys.stdout.flush()
            if result[-1] == 0:
                return False
            if result[-1] > self.local_last_step:
                self.local_last_step = result[-1]
            else:
                logger(
                    "global_step not updating during {} test steps".format(
                        self.during_step
                    )
                )
                sys.stdout.flush()
                version = tf.VERSION.split(".")
                if int(version[0]) == 1 and int(version[1]) <= 8:
                    sess.request_stop(notify_all=True)
                return True
        self.local_test_step += 1
        return False

    def _inference(self, sess, *args, **kwargs):
        feed_dict = {"training:0": False}
        result = sess.run(self._inference_ops + self._evaluate_ops, feed_dict=feed_dict)
        if self.FLAGS.outputs is not None and self.odps_writer is None:
            self.odps_writer = tf.python_io.TableWriter(
                self.FLAGS.outputs, slice_id=self.FLAGS.task_index
            )
        values = []
        values = []
        for i in range(len(result[0])):
            values.append(
                (
                    str(result[0][i][0]),
                    int(result[1][i][0]),
                    float(result[2][i][0]),
                    int(result[3][i][0]),
                    float(result[4][i][0]),
                    # int(result[4][i][0]), float(result[5][i][0]),
                    float(result[5][i][0]),
                )
            )
        self.odps_writer.write(values, indices=range(len(values[0])))
        if self.local_predict_step % self.during_step == 0:
            print("Predict step {}: {}".format(self.local_predict_step, values[0]))
            logger(
                self.table_progress(
                    self.local_predict_step * self.FLAGS.batch_size,
                    self.table_row_count,
                )
            )
            # sys.stdout.flush()
        self.local_predict_step += 1
        from tensorflow.python.platform import gfile

        matadata_name = "matadata.tsv"
        print(" |--write matedata to =", matadata_name)
        with gfile.GFile(matadata_name, "w") as f:
            print(result[1])
            for vid_labels in result[1]:
                f.write("\t".join(vid_labels) + "\n")
        return False


class Runner(BaseRunner):
    def __init__(self, FLAGS, *args, **kwargs):
        super(Runner, self).__init__(FLAGS, *args, **kwargs)
        self.save_summaries_steps = FLAGS.save_summaries_steps
        self.local_summary_step = 0
        self.summary_op = None
        self.summary_writer = None
        self.ctr_auc = 0
        self.cvr_auc = 0
        self.ctcvr_auc = 0

        if FLAGS.task_index == 0:
            self.summary_writer = tf.summary.MetricsWriter("./logs/")

    def set_summary_op(self, op):
        self.summary_op = op
    
    
    def _train(self, sess, model, *args, **kwargs):
        print("Runner._train: self.local_train_step: ", self.local_train_step)
        print("Runner._train: tf.train.get_global_step: ", tf.train.get_global_step(), tf.train.get_global_step().type)
        
        p = float(self.local_train_step)/tf.train.get_global_step()
        l = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 0.01 / (1. + 10 * p)**0.75
        feed_dict = {"training:0": True, model.head_model.l: l} 
        
        # feed_dict = {"training:0": True, }
        result = sess.run(self._log_ops[1] + self._train_ops + [tf.train.get_global_step()], feed_dict=feed_dict)
        if self.local_train_step % self.during_step == 0:
            log_entries = []
            for k, v in zip(self._log_ops[0], result):
                if isinstance(v, (float, int)):
                    log_entries.append("{}={:.7g}".format(k, v))
                else:
                    log_entries.append("{}={}".format(k, str(v)))
            logger("Train:{}".format(", ".join(log_entries)))
            sys.stdout.flush()
        self.local_train_step += 1
        return False

    def _evaluate(self, sess, *args, **kwargs):
        feed_dict = {"training:0": False}
        result = sess.run(
            self._log_ops[1]
            + self._evaluate_ops
            + [self.summary_op[1], tf.train.get_global_step()],
            feed_dict=feed_dict,
            options=self.run_options,
            run_metadata=self.run_metadata,
        )
        if self.FLAGS.task_index == 0 and self.FLAGS.timeline:
            timeline_trace = timeline.Timeline(step_stats=self.run_metadata.step_stats)
            chrome_trace = timeline_trace.generate_chrome_trace_format()
            # f.write(chrome_trace)
            file_io.write_string_to_file(
                os.path.join(self.FLAGS.checkpointDir, "timeline.json"), chrome_trace,
            )
        self.ctr_auc = result[0]
        self.cvr_auc = result[1]
        # 添加summary
        # summary_op:{name:tensor},result[-2]:
        # print(self.summary_op[0])
        if not self.model_start_flag:
            self.original_global_step = result[-1]
            print("original global step:{}".format(self.original_global_step))
            self.start_time = time.time()
            self.model_start_flag = 1
        if self.summary_writer is not None:
            for name, value in zip(self.summary_op[0], result[-2]):
                self.summary_writer.add_scalar(name, value, self.local_test_step)
        if self.local_test_step % self.during_step == 0:
            log_entries = []
            for k, v in zip(self._log_ops[0], result):
                if isinstance(v, (float, int)):
                    log_entries.append("{}={:.7g}".format(k, v))
                else:
                    log_entries.append("{}={}".format(k, str(v)))
            logger("Validation:{}".format(", ".join(log_entries)))
            logger(self.table_progress(self.local_test_step * self.FLAGS.batch_size, self.table_row_count))
            sys.stdout.flush()  
        # 如果global_step还是初始值，说明其他worker没有开始训练，不需要停止
        if result[-1] == self.original_global_step:
            if self.local_test_step % self.during_step == 0:
                # 如果训练worker过快，导致chief刚开始其他worker就结束了，判断一直是初始值超过十分钟，就突出
                if time.time() - self.start_time > 60 * 10:
                    print("global step is initial value over 10min")
                    return True
                print("global step is initial value")
        # 如果global_step不是初始值，但是大于local_last_step，说明其他worker已经开始训练，不需要停止，并且更新local_last_step
        elif result[-1] > self.local_last_step:
            if self.local_test_step % self.during_step == 0:
                print("global step is updating...")
            self.local_last_step = result[-1]
        # 如果启动时间不超过五分钟，不需要停止
        elif time.time() - self.start_time < 300:
            if self.local_test_step % self.during_step == 0:
                print("time not enough")
        # 如果global_step不是初始值，但是小于等于local_last_step，说明其他worker已经停止训练，需要停止，或者其他worker过慢
        elif self.local_test_step % self.during_step == 0:
            logger(
                "global_step not updating during {} test steps, now global_step{}".format(
                    self.during_step, result[-1]
                )
            )
            sys.stdout.flush()
            version = tf.VERSION.split(".")
            # 1.4.0,1.12.0,2.0.0
            if int(version[0]) == 1 and int(version[1]) <= 8:
                sess.request_stop(notify_all=True)
            return True

        self.local_test_step += 1
        return False

    def _inference(self, sess, *args, **kwargs):
        feed_dict = {"training:0": False}
        result = sess.run(
            self._inference_ops + self._log_ops[1] + self._evaluate_ops,
            feed_dict=feed_dict,
            options=self.run_options,
            run_metadata=self.run_metadata,
        )
        # sys.stdout.flush()
        if self.FLAGS.outputs is not None and self.odps_writer is None:
            self.odps_writer = tf.python_io.TableWriter(
                self.FLAGS.outputs, slice_id=self.FLAGS.task_index
            )
        output_table_cols = len(self._inference_ops)
        values = []
        for i in range(len(result[0])):
            values.append(
                (
                    str(result[0][i][0]),
                    int(result[1][i][0]),
                    float(result[2][i][0]),
                    int(result[3][i][0]),
                    float(result[4][i][0]),
                    # int(result[4][i][0]), float(result[5][i][0]),
                    float(result[5][i][0]),
                )
                + tuple([str(result[j][i][0]) for j in range(6, output_table_cols)])
            )

        self.odps_writer.write(values, indices=range(len(values[0])))
        if self.local_predict_step % self.during_step == 0:
            if self.FLAGS.task_index == 0 and self.FLAGS.timeline:
                timeline_trace = timeline.Timeline(
                    step_stats=self.run_metadata.step_stats
                )
                chrome_trace = timeline_trace.generate_chrome_trace_format()
                # f.write(chrome_trace)
                file_io.write_string_to_file(
                    os.path.join(self.FLAGS.checkpointDir, "timeline.json"),
                    chrome_trace,
                )
            print("result[-1]", result[-1])
            print("Predict step {}: {}".format(self.local_predict_step, values[0]))
            
            log_entries = []
            for k, v in zip(self._log_ops[0], result[len(self._inference_ops):len(self._inference_ops) + len(self._log_ops[0])]):
                if isinstance(v, (float, int)):
                    log_entries.append("{}={:.7g}".format(k, v))
                else:
                    log_entries.append("{}={}".format(k, str(v)))
            logger("Inference:{}".format(", ".join(log_entries)))
            logger(
                self.table_progress(
                    self.local_predict_step * self.FLAGS.batch_size,
                    self.table_row_count,
                )
            )
            # sys.stdout.flush()
        self.local_predict_step += 1
        return False


class RestoreRunner(BaseRunner):
    def __init__(self, FLAGS, *args, **kwargs):
        super(RestoreRunner, self).__init__(FLAGS, *args, **kwargs)
        self.save_summaries_steps = FLAGS.save_summaries_steps
        self.local_summary_step = 0
        self.summary_op = None
        self.summary_writer = None
        self.ctr_auc = 0
        self.cvr_auc = 0
        self.ctcvr_auc = 0

        if FLAGS.task_index == 0:
            self.summary_writer = tf.summary.MetricsWriter("./")

    def set_summary_op(self, op):
        self.summary_op = op

    def _train(self, sess, *args, **kwargs):
        feed_dict = {"training:0": True}

        result = sess.run(
            self._log_ops[1] + self._train_ops + [tf.train.get_global_step()],
            feed_dict=feed_dict,
        )

        # print debug ops
        if self._debug_ops is not None:
            debug_result = sess.run(self._debug_ops[1], feed_dict=feed_dict)
            for name, value in zip(self._debug_ops[0], debug_result):
                print(name, value)
        if self.local_train_step % self.during_step == 0:
            logger(
                "Train:{}".format(
                    ", ".join(
                        [
                            "{}={:.7g}".format(k, v)
                            for k, v in zip(self._log_ops[0], result)
                        ]
                    )
                )
            )
            logger(
                self.table_progress(
                    self.local_train_step * self.FLAGS.batch_size, self.table_row_count
                )
            )
            sys.stdout.flush()
        self.local_train_step += 1
        return False

    def _evaluate(self, sess, *args, **kwargs):
        feed_dict = {"training:0": False}
        result = sess.run(
            self._log_ops[1]
            + self._evaluate_ops
            + [self.summary_op[1], tf.train.get_global_step()],
            feed_dict=feed_dict,
        )
        self.ctr_auc = result[0]
        self.cvr_auc = result[1]
        if not self.model_start_flag:
            self.original_global_step = result[-1]
            print("original global step:{}".format(self.original_global_step))
            self.start_time = time.time()
            self.model_start_flag = 1
        if self.summary_writer is not None:
            for name, value in zip(self.summary_op[0], result[-2]):
                self.summary_writer.add_scalar(name, value, self.local_test_step)
        if self.local_test_step % self.during_step == 0:
            logger(
                "Validation:{}".format(
                    ", ".join(
                        [
                            "{}={:.7g}".format(k, v)
                            for k, v in zip(self._log_ops[0], result)
                        ]
                    )
                )
            )
            logger(
                self.table_progress(
                    self.local_test_step * self.FLAGS.batch_size, self.table_row_count
                )
            )
            sys.stdout.flush()
        # 如果global_step还是初始值，说明其他worker没有开始训练，不需要停止
        if result[-1] == self.original_global_step:
            if self.local_test_step % self.during_step == 0:
                # 如果训练worker过快，导致chief刚开始其他worker就结束了，判断一直是初始值超过十分钟，就突出
                if time.time() - self.start_time > 60 * 10:
                    print("global step is initial value over 10min")
                    return True
                print("global step is initial value")
        # 如果global_step不是初始值，但是大于local_last_step，说明其他worker已经开始训练，不需要停止，并且更新local_last_step
        elif result[-1] > self.local_last_step:
            if self.local_test_step % self.during_step == 0:
                print("global step is updating...")
            self.local_last_step = result[-1]
        # 如果启动时间不超过五分钟，不需要停止
        elif time.time() - self.start_time < 300:
            if self.local_test_step % self.during_step == 0:
                print("time not enough")
        # 如果global_step不是初始值，但是小于等于local_last_step，说明其他worker已经停止训练，需要停止，或者其他worker过慢
        elif self.local_test_step % self.during_step == 0:
            logger(
                "global_step not updating during {} test steps, now global_step{}".format(
                    self.during_step, result[-1]
                )
            )
            sys.stdout.flush()
            version = tf.VERSION.split(".")
            # 1.4.0,1.12.0,2.0.0
            if int(version[0]) == 1 and int(version[1]) <= 8:
                sess.request_stop(notify_all=True)
            return True

        self.local_test_step += 1
        return False

    def _inference(self, sess, *args, **kwargs):
        feed_dict = {"training:0": False}
        result = sess.run(
            self._inference_ops + self._log_ops[1] + self._evaluate_ops,
            feed_dict=feed_dict,
        )
        # sys.stdout.flush()
        if self.FLAGS.outputs is not None and self.odps_writer is None:
            self.odps_writer = tf.python_io.TableWriter(
                self.FLAGS.outputs, slice_id=self.FLAGS.task_index
            )
        output_table_cols = len(self._inference_ops)
        values = []
        for i in range(len(result[0])):
            values.append(
                (
                    str(result[0][i][0]),
                    int(result[1][i][0]),
                    float(result[2][i][0]),
                    int(result[3][i][0]),
                    float(result[4][i][0]),
                    # int(result[4][i][0]), float(result[5][i][0]),
                    float(result[5][i][0]),
                )
                + tuple([str(result[j][i][0]) for j in range(6, output_table_cols)])
            )

        self.odps_writer.write(values, indices=range(len(values[0])))
        if self.local_predict_step % self.during_step == 0:
            print("result[-1]", result[-1])
            print("Predict step {}: {}".format(self.local_predict_step, values[0]))
            metrics_list = [
                [k, "{:.7g}".format(v)]
                for k, v in zip(
                    self._log_ops[0],
                    result[
                        len(self._inference_ops) : len(self._inference_ops)
                        + len(self._log_ops[0])
                    ],
                )
            ]
            print(tabulate(metrics_list, headers=["Metric", "Value"], tablefmt="grid"))
            logger(
                self.table_progress(
                    self.local_predict_step * self.FLAGS.batch_size,
                    self.table_row_count,
                )
            )
            return True
            # sys.stdout.flush()
        self.local_predict_step += 1
        return False


class Runner_emb(BaseRunner):
    def __init__(self, FLAGS, *args, **kwargs):
        super(Runner_emb, self).__init__(FLAGS, *args, **kwargs)
        self.save_summaries_steps = FLAGS.save_summaries_steps
        self.local_summary_step = 0
        self.summary_op = None
        self.summary_writer = None
        self.ctr_auc = 0
        self.cvr_auc = 0
        self.ctcvr_auc = 0

        if FLAGS.task_index == 0:
            self.summary_writer = tf.summary.MetricsWriter("./")

    def set_summary_op(self, op):
        self.summary_op = op

    def _train(self, sess, *args, **kwargs):
        feed_dict = {"training:0": True}

        result = sess.run(
            self._log_ops[1] + self._train_ops + [tf.train.get_global_step()],
            feed_dict=feed_dict,
        )

        # print debug ops
        if self._debug_ops is not None:
            debug_result = sess.run(self._debug_ops[1], feed_dict=feed_dict)
            for name, value in zip(self._debug_ops[0], debug_result):
                print(name, value)
        if self.local_train_step % self.during_step == 0:
            logger(
                "Train:{}".format(
                    ", ".join(
                        [
                            "{}={:.7g}".format(k, v)
                            for k, v in zip(self._log_ops[0], result)
                        ]
                    )
                )
            )
            logger(
                self.table_progress(
                    self.local_train_step * self.FLAGS.batch_size, self.table_row_count
                )
            )
            sys.stdout.flush()
        self.local_train_step += 1
        return False

    def _evaluate(self, sess, *args, **kwargs):
        feed_dict = {"training:0": False}
        result = sess.run(
            self._log_ops[1]
            + self._evaluate_ops
            + [self.summary_op[1], tf.train.get_global_step()],
            feed_dict=feed_dict,
        )
        self.ctr_auc = result[0]
        self.cvr_auc = result[1]
        # 添加summary
        # summary_op:{name:tensor},result[-2]:
        # print(self.summary_op[0])
        if not self.model_start_flag:
            self.original_global_step = result[-1]
            print("original global step:{}".format(self.original_global_step))
            self.start_time = time.time()
            self.model_start_flag = 1
        if self.summary_writer is not None:
            for name, value in zip(self.summary_op[0], result[-2]):
                self.summary_writer.add_scalar(name, value, self.local_test_step)
        if self.local_test_step % self.during_step == 0:
            logger(
                "Validation:{}".format(
                    ", ".join(
                        [
                            "{}={:.7g}".format(k, v)
                            for k, v in zip(self._log_ops[0], result)
                        ]
                    )
                )
            )
            logger(
                self.table_progress(
                    self.local_test_step * self.FLAGS.batch_size, self.table_row_count
                )
            )
            sys.stdout.flush()
        # 如果global_step还是初始值，说明其他worker没有开始训练，不需要停止
        if result[-1] == self.original_global_step:
            if self.local_test_step % self.during_step == 0:
                print("global step is initial value")
        # 如果global_step不是初始值，但是大于local_last_step，说明其他worker已经开始训练，不需要停止，并且更新local_last_step
        elif result[-1] > self.local_last_step:
            if self.local_test_step % self.during_step == 0:
                print("global step is updating...")
            self.local_last_step = result[-1]
        # 如果启动时间不超过五分钟，不需要停止
        elif time.time() - self.start_time < 300:
            if self.local_test_step % self.during_step == 0:
                print("time not enough")
        # 如果global_step不是初始值，但是小于等于local_last_step，说明其他worker已经停止训练，需要停止
        else:
            logger(
                "global_step not updating during {} test steps, now global_step{}".format(
                    self.during_step, result[-1]
                )
            )
            sys.stdout.flush()
            version = tf.VERSION.split(".")
            # 1.4.0,1.12.0,2.0.0
            if int(version[0]) == 1 and int(version[1]) <= 8:
                sess.request_stop(notify_all=True)
            return True
        self.local_test_step += 1
        return False

    def _inference(self, sess, *args, **kwargs):
        feed_dict = {"training:0": False}
        result = sess.run(self._inference_ops, feed_dict=feed_dict)
        # sys.stdout.flush()
        if self.FLAGS.outputs is not None and self.odps_writer is None:
            self.odps_writer = tf.python_io.TableWriter(
                self.FLAGS.outputs, slice_id=self.FLAGS.task_index
            )
        values = []
        # 因为涉及到向量的打印，因此禁止打印
        # print("result[-1]", result[-1])
        for i in range(len(result[0])):
            values.append(
                (
                    str(result[0][i][0]),
                    int(result[1][i][0]),
                    float(result[2][i][0]),
                    int(result[3][i][0]),
                    float(result[4][i][0]),
                    # int(result[4][i][0]), float(result[5][i][0]),
                    float(result[5][i][0]),
                    str(result[6][i]),
                    str(result[7][i]),
                )
            )
        self.odps_writer.write(values, indices=range(len(values[0])))
        if self.local_predict_step % self.during_step == 0:
            print("Predict step {}: {}".format(self.local_predict_step, values[0]))
            logger(
                self.table_progress(
                    self.local_predict_step * self.FLAGS.batch_size,
                    self.table_row_count,
                )
            )
            # sys.stdout.flush()
        self.local_predict_step += 1
        return False


class LocalRunner(BaseRunner):
    def __init__(self, FLAGS, *args, **kwargs):
        super(LocalRunner, self).__init__(FLAGS, *args, **kwargs)
        self.save_summaries_steps = FLAGS.save_summaries_steps
        self.local_summary_step = 0
        self.summary_op = None

    def set_summary_op(self, op):
        self.summary_op = op

    def _train(self, sess, *args, **kwargs):
        feed_dict = {"training:0": True}

        result = sess.run(
            self._log_ops[1] + self._train_ops + [tf.train.get_global_step()],
            feed_dict=feed_dict,
        )
        # print debug ops
        if self._debug_ops is not None:
            debug_result = sess.run(self._debug_ops[1], feed_dict=feed_dict)
            for name, value in zip(self._debug_ops[0], debug_result):
                print(name, value)
        if self.local_train_step % self.during_step == 0:
            logger(
                "Train:{}".format(
                    ", ".join(
                        [
                            "{}={:.7g}".format(k, v)
                            for k, v in zip(self._log_ops[0], result)
                        ]
                    )
                )
            )
            sys.stdout.flush()
        self.local_train_step += 1
        return False

    def _evaluate(self, sess, *args, **kwargs):
        feed_dict = {"training:0": False}
        result = sess.run(
            self._log_ops[1] + self._evaluate_ops + [tf.train.get_global_step()],
            feed_dict=feed_dict,
        )
        # 添加summary
        # summary_op:{name:tensor},result[-2]:
        if self.summary_writer is not None:
            for name, value in zip(self._log_ops[0], result):
                self.summary_writer.add_scalar(name, value, self.local_test_step)
        if self.local_test_step % self.during_step == 0:
            logger(
                "Validation:{}".format(
                    ", ".join(
                        [
                            "{}={:.7g}".format(k, v)
                            for k, v in zip(self._log_ops[0], result)
                        ]
                    )
                )
            )
            sys.stdout.flush()
            if result[-1] > self.local_last_step:
                self.local_last_step = result[-1]
            else:
                logger(
                    "global_step not updating during {} test steps".format(
                        self.during_step
                    )
                )
                sys.stdout.flush()
                version = tf.VERSION.split(".")
                # 1.4.0,1.12.0,2.0.0
                if int(version[0]) == 1 and int(version[1]) <= 8:
                    sess.request_stop(notify_all=True)
                return True
        self.local_test_step += 1
        return False

    def run(self, sess, *args, **kwargs):
        if self.FLAGS.mode_run == ModeKeys.LOCAL:
            return self._train(sess)
        elif self.FLAGS.mode_run == ModeKeys.PREDICT:
            return self._inference(sess)
        elif self.FLAGS.mode_run == ModeKeys.TRAIN:
            return self._train(sess)
        elif self.FLAGS.mode_run == ModeKeys.EVAL:
            return self._evaluate(sess)
        else:
            raise Exception("Unsupported Mode:{}".format(self.FLAGS.mode_run))
