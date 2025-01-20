import tensorflow as tf
import importlib

flags = tf.app.flags
flags.DEFINE_string('tables', None, 'input')
flags.DEFINE_string('outputs', None, 'output')
flags.DEFINE_string('fg_conf', None, 'fg.json')
flags.DEFINE_string('global_conf', None, 'global conf')
flags.DEFINE_string('model_name', None, 'model name')
flags.DEFINE_string('pai_file_name', None, 'pai_file_name')
flags.DEFINE_string('checkpointDir', None, 'Summaries log directory')
flags.DEFINE_string('version_dir', None, 'path to version directory')
flags.DEFINE_integer('save_checkpoint_secs', 36, 'The frequency with which summaries are saved, in seconds.')
flags.DEFINE_integer('save_summaries_secs', 300, 'The frequency with which summaries are saved, in seconds.')
flags.DEFINE_integer('save_summaries_steps', 500, 'The frequency with which logs are print.')
flags.DEFINE_integer('print_steps', 100, 'The frequency with which logs are print to logview.')
flags.DEFINE_integer('batch_size', 128, 'batch size.')
flags.DEFINE_integer('num_epochs', 1, 'num epochs.')
flags.DEFINE_string('test_params', '', 'params for test')
flags.DEFINE_integer('max_checkpoints_to_keep', 1, 'max checkpoints, in Train.')
flags.DEFINE_integer('is_predict', 0, 'Predict or Train.')
flags.DEFINE_string('mode_run', 'train', 'predict or train.')
flags.DEFINE_integer('exclude_click', 0, 'click as negative')
flags.DEFINE_string('ps_hosts', '', 'ps hosts')
flags.DEFINE_string('worker_hosts', '', 'worker hosts')
flags.DEFINE_string('job_name', '', 'job name: worker or ps')
flags.DEFINE_integer('task_index', 0, 'Worker task index')

flags.DEFINE_string('buckets', None, 'checkpoint pre store dir')
flags.DEFINE_string('restore_dir', None, 'checkpoint pre store dir')
flags.DEFINE_integer('restore_flag', 0, 'restore_flag')

# for export emb
flags.DEFINE_integer('export_emb_flag', 0, 'export_emb_flag')
flags.DEFINE_string('export_tensor_names', '', 'params for test')
flags.DEFINE_string('export_op_names', '', 'params for test')

# for pairwise loss wieght
flags.DEFINE_float('pairwise_loss_weight', 0.2, 'pairwise loss weight.')

flags.DEFINE_integer('inter_op_parallelism_threads', 8, 'inter_op_parallelism_threads')
flags.DEFINE_integer('intra_op_parallelism_threads', 8, 'intra_op_parallelism_threads')

flags.DEFINE_integer('prefetch', 0, 'prefetch')
flags.DEFINE_integer('prefetch_capacity', 4, 'prefetch_capacity')
flags.DEFINE_integer('prefetch_num_threads', 4, 'prefetch_num_threads')
flags.DEFINE_string('vit_hyp', 'patch_size=8,num_layers=2,num_features=32,num_heads=4,mlp_dim=128,image_l=50,image_w=50', 'vit')



# dataset type
flags.DEFINE_string('dataset_type', 'v2', 'dataset_type')

flags.DEFINE_integer('timeline', 0, 'timeline_flag')


FLAGS = tf.app.flags.FLAGS


def main(_):
    
    available_device_path = '/job:localhost/replica:0/task:0/device:CPU:0'
    with tf.device(available_device_path):
        importlib.import_module('%s' % FLAGS.pai_file_name).run(FLAGS)
    # print("pai_file_name: %s" % FLAGS.pai_file_name)
    # importlib.import_module('%s' % FLAGS.pai_file_name).run(FLAGS)


if __name__ == '__main__':
    tf.app.run()
