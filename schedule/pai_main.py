# encoding: utf-8
from __future__ import print_function
from utils import util
from tensorflow.python.lib.io import file_io
import os
import sys
import json
import time
import random
import importlib
import numpy as np
import tensorflow as tf
import pai
from model_ops.tflog import tflogger as logging
from utils.dingding import dingding


# tf.logging.set_verbosity(tf.logging.WARN)
currentPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(currentPath + os.sep + "../")
sys.path.append(currentPath + os.sep + "../..")

tf.set_random_seed(2022)
np.random.seed(2022)
random.seed(2022)


def get_now_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def get_model(global_conf, FLAGS):
    print("import model: %s" % FLAGS.model_name)
    return importlib.import_module("models.%s" % FLAGS.model_name).DNNModel(
        global_conf, FLAGS
    )


def prepare_default():
    return [
        [tf.constant("", dtype=tf.string)],
        [tf.constant("0;0;0;0", dtype=tf.string)],
        [tf.constant("", dtype=tf.string)],
    ]


def table_progress(done, all):
    return format(float(all - done) / all * 100, "0.0f")

def get_batch_data_v2(
    tables, batch_size, slice_count, slice_id, num_threads=None, num_epochs=1, name=""
):
    dataset = tf.data.TableRecordDataset(
        tables,
        record_defaults=("", "", ""),
        selected_cols="sample_id,labels,features_kv",
        slice_id=slice_id,
        slice_count=slice_count,
        num_threads=num_threads,
        capacity=None,
    )
    dataset = dataset.shuffle(buffer_size=batch_size * 3, reshuffle_each_iteration=True,seed=2022)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size).prefetch(1)
    batch_data = dataset.make_one_shot_iterator().get_next()
    return batch_data

def run(FLAGS):
    # Predict or Train
    is_predict = FLAGS.mode_run == "predict"
    is_chief = FLAGS.task_index == 0
    print("is_predict = %s" % is_predict)

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index, protocol="grpc++"
    )

    worker_count = len(worker_hosts)
    print("job name = %s" % FLAGS.job_name)
    print("worker count ", worker_count)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device("/job:worker/task:%d/cpu:0" % FLAGS.task_index):
            print("task index = %d" % FLAGS.task_index)
            print("tables: " + FLAGS.tables)
            tables = FLAGS.tables.split(",")
            if is_predict:
                table, slice_count, slice_id, name = (
                    tables[-1:],
                    worker_count,
                    FLAGS.task_index,
                    "predict",
                )
            elif is_chief:
                table, slice_count, slice_id, name = tables[-1:], 1, 0, "chief"
            else:
                table, slice_count, slice_id, name = (
                    tables[0:-1],
                    worker_count - 1,
                    FLAGS.task_index - 1,
                    "not_chief",
                )
            if FLAGS.dataset_type=="v2":
                batch_data = get_batch_data_v2(
                    table,
                    batch_size=FLAGS.batch_size,
                    slice_count=slice_count,
                    slice_id=slice_id,
                    name=name,
                )
            reader = tf.python_io.TableReader(
                table[0], slice_count=slice_count, slice_id=slice_id
            )
            table_row_count = reader.get_row_count()*len(table)
            print("table rows:", table_row_count)
        # Assigns ops to the local worker by default.
        with tf.device(
            tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster
            )
        ):
            print("currentPath: ", currentPath)
            with open(currentPath + "/../conf/" + FLAGS.global_conf, "r") as load_f:
                global_conf = json.load(load_f)

            # update param
            if len(FLAGS.test_params) > 0:
                test_params = util.string2kv(FLAGS.test_params, "&", "=")
                for k, v in test_params.items():
                    if k in global_conf["model"]:
                        new_v = float(test_params[k])
                        if k == "need_dropout":
                            if new_v == 1:
                                new_v = True
                            else:
                                new_v = False
                        print(
                            "Model_Config update {} from {} to {}".format(
                                k, global_conf["model"][k], new_v
                            )
                        )
                        global_conf["model"].update({k: new_v})
            global_conf["fg_path"] = currentPath + "/../conf/" + FLAGS.fg_conf

            model = get_model(global_conf, FLAGS)
            model.build(batch_data)
            # merged_summary = model.summary()

        conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        conf.gpu_options.allow_growth = True
        allowed_devices = [
            "/job:ps/replica:0/task:0/device:CPU:0",
            "/job:ps/replica:0/task:0/device:XLA_CPU:0",
            "/job:worker/replica:0/task:0/device:CPU:0",
            "/job:worker/replica:0/task:0/device:XLA_CPU:0",
            "/job:worker/replica:0/task:1/device:CPU:0",
            "/job:worker/replica:0/task:1/device:XLA_CPU:0",
            "/job:worker/replica:0/task:2/device:CPU:0",
            "/job:worker/replica:0/task:2/device:XLA_CPU:0",
            "/job:worker/replica:0/task:3/device:CPU:0",
            "/job:worker/replica:0/task:3/device:XLA_CPU:0"
        ]
        conf.device_filters.extend(allowed_devices)

        version = tf.VERSION.split(".")
        # 1.4.0,1.12.0,2.0.0
        print("conf:",conf)
        print("tf.VERSION:",tf.VERSION)
        print("version:{}.{}".format(version[0],version[1]))
        print("conf.inter_op_parallelism_threads:",conf.inter_op_parallelism_threads)
        print("conf.intra_op_parallelism_threads:",conf.intra_op_parallelism_threads)
        if not (int(version[0]) == 1 and int(version[1]) <= 8):
            conf.inter_op_parallelism_threads = FLAGS.inter_op_parallelism_threads
            conf.intra_op_parallelism_threads = FLAGS.intra_op_parallelism_threads
        
        if FLAGS.task_index == 0:
            with open(global_conf["fg_path"]) as f:
                fg_conf = json.load(f)
            file_io.write_string_to_file(
                os.path.join(FLAGS.checkpointDir, "fg.json"),
                json.dumps(fg_conf, indent=2),
            )
        logging.info("task=%d starts" % FLAGS.task_index)
        from model_ops.restorer import get_restore

        # saver, scaffold, ckpt_path = get_restore(FLAGS)
        saver, scaffold, ckpt_path = None, None, None
        hooks = []
        if FLAGS.prefetch:
            hooks.append(pai.data.make_prefetch_hook())

        from utils.hook import ChiefLastExitHook
        chief_last_exit_hook = ChiefLastExitHook(worker_count, is_chief)
        hooks.append(chief_last_exit_hook)
        
        
        print("main:saver:",saver)
        print("main:scaffold:",scaffold)
        print("main:ckpt_path:",ckpt_path)
        
        # try:
        with tf.device(tf.train.replica_device_setter(ps_tasks=4)):    
            with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=is_chief,
                checkpoint_dir=FLAGS.checkpointDir,
                save_checkpoint_secs=FLAGS.save_checkpoint_secs,
                save_summaries_steps=FLAGS.save_summaries_steps,
                hooks=hooks,
                scaffold=scaffold,
                config=conf,
            ) as mon_sess:
                if saver and FLAGS.task_index != 0:
                    logging.info("start to restore")
                    saver.restore(mon_sess, ckpt_path)
                    logging.info("restore checkpoint: done.")
                mon_sess.graph.finalize()
                try:
                    if FLAGS.task_index == 0:
                        dingding("train:model【{}】start【{}】at {}".format(FLAGS.model_name,FLAGS.mode_run,get_now_time()))
                        model_start_time = time.time()
                except Exception as e:
                    print("dingding error: ", e)
                try:
                    if not mon_sess.should_stop():
                        model.runner.before_run(mon_sess)
                    while not mon_sess.should_stop():
                        # if model.runner.run(mon_sess, table_row_count):
                        if model.runner.run(mon_sess, model, table_row_count):
                            labels_tensor = mon_sess.graph.get_tensor_by_name('dann_pvpay_Input_Pipeline/Maximum_2:0')
                            labels_value = mon_sess.run(labels_tensor, feed_dict={model.is_training: True})
                            print("Labels value:", labels_value)
                            
                            predictions = mon_sess.graph.get_tensor_by_name('Predict/Sigmoid:0')
                            # 使用 feed_dict 为 training 占位符提供一个布尔值，例如 True
                            predictions_value = mon_sess.run(predictions, feed_dict={model.is_training: True})
                            # predictions_value = mon_sess.run(predictions, feed_dict={'training': True})
                            print("predictions value:", predictions_value)
                            break
                        local_train_step += 1
                        
                except tf.errors.OutOfRangeError:
                    print("Run out of data.")
                    version = tf.VERSION.split(".")
                    if int(version[0]) == 1 and int(version[1]) <= 8:
                        mon_sess.request_stop(notify_all=True)
                finally:
                    if not mon_sess.should_stop():
                        model.runner.after_run(mon_sess)
                    try:
                        if FLAGS.task_index == 0:
                            dingding(
                                "train:model【{}】finish【{}】at {},ctr_auc:xx vs {},cvr_auc:xx vs {}, cost {}".format(
                                    FLAGS.model_name,
                                    FLAGS.mode_run,
                                    get_now_time(),
                                    model.runner.ctr_auc,
                                    model.runner.cvr_auc,
                                    time.strftime("%H:%M:%S", time.gmtime(time.time() - model_start_time))
                                )
                            )
                    except Exception as e:
                        print("dingding error: ", e)
                auc_threshold = 0
                if not is_predict and is_chief and model.runner.ctr_auc < auc_threshold:
                    raise Exception('auc is less than threshold,{}<{}'.format(model.runner.ctr_auc, auc_threshold))
                print("Finish.")
                time.sleep(15)
       
