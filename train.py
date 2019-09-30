"""Trains the AudioUNet Model"""
import tensorflow as tf

from tfnet import TFNetEstimator
from tfnet import nets
from tfnet import summaries
import argshelper

import datahelper.dataset as ds
FLAGS = argshelper.FLAGS

def main(argv):
    if FLAGS.debug:
        print("Unprocessed flags:", argv)
        tf.logging.set_verbosity(tf.logging.DEBUG)
        tf.logging.debug('-------------------------------------------')
        tf.logging.debug('DEBUG MODE')
        tf.logging.debug('-------------------------------------------')

        tf.logging.debug('Time params:' + str(argshelper.get_time_params()))
        tf.logging.debug('Freq params:' + str(argshelper.get_freq_params()))




    tf.logging.set_verbosity(tf.logging.INFO)
    degrade_fn = lambda x: ds.downsample_by(x, FLAGS.downsample_rate)
    dset = ds.get_dataset(FLAGS.trainset,
                          path=FLAGS.datapath,
                          degrade_fn=degrade_fn,
                          epochs=FLAGS.epochs,
                          batchsize=FLAGS.batchsize,
                          segs_per_sample=FLAGS.batchsize//4,
                         )
    #train_input_fn = lambda: dset().make_one_shot_iterator().get_next()
    train_input_fn = dset
    if FLAGS.testset:
        eval_dset = ds.get_dataset(FLAGS.testset,
                                   path=FLAGS.datapath,
                                   epochs=1,
                                   degrade_fn=degrade_fn,
                                   batchsize=FLAGS.batchsize,
                                   shuffle=False
                                  )
        #eval_input_fn = lambda: eval_dset().make_one_shot_iterator().get_next()
        eval_input_fn = eval_dset

    if FLAGS.multigpu:
        config = argshelper.distribute.multi_gpu_config(
            log_step_count_steps=FLAGS.log_step_count_steps)
    else:
        config = tf.estimator.RunConfig(
            log_step_count_steps=FLAGS.log_step_count_steps)

    if FLAGS.save_checkpoints_steps:
        config = config.replace(save_checkpoints_steps=FLAGS.save_checkpoints_steps)

    if FLAGS.save_checkpoints_secs:
        config = config.replace(save_checkpoints_secs=FLAGS.save_checkpoints_secs)

    if FLAGS.usexla:
        sess_config = tf.ConfigProto()
        sess_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        config = config.replace(session_config=sess_config)

    _summaries = []
    if FLAGS.audio_sample_rate > 0:
        _summaries.append(summaries.audio_sample_summary(FLAGS.audio_sample_rate))
    elif FLAGS.audio_sample_rate == 0:
        raise NotImplementedError('Automatic sample rate determination is not'
                                  'yet implemented')
    _summaries.append(summaries.audio_spectrogram_summary())

    if FLAGS.spectral_copies:
        net_config = nets.build_net(FLAGS.objective, FLAGS.downsample_rate,
                                    time_params=argshelper.get_time_params(),
                                    freq_params=argshelper.get_freq_params(),
                                    window_length=FLAGS.window_length,
                                    transform=FLAGS.transform,
                                    fusion_op=FLAGS.fusion_op,
                                   )
    else:
        net_config = nets.build_net(FLAGS.objective,
                                    time_params=argshelper.get_time_params(),
                                    freq_params=argshelper.get_freq_params(),
                                    window_length=FLAGS.window_length,
                                    transform=FLAGS.transform,
                                    fusion_op=FLAGS.fusion_op,
                                   )

    if FLAGS.learning_rate_decay:
        learning_rate = lambda: tf.train.polynomial_decay(FLAGS.learning_rate,
                                                          end_learning_rate=1e-6,
                                                          global_step=tf.train.get_global_step(),
                                                          decay_steps=500000,
                                                          power=0.5)
    else:
        learning_rate = lambda: FLAGS.learning_rate


    optimizers = {'adam': lambda: tf.train.AdamOptimizer(learning_rate=learning_rate()),
                  'sgd': lambda: tf.train.GradientDescentOptimizer(learning_rate=learning_rate()),
                 }
    
    print('------------------------------------------')
    print('Save steps: {}, ({}s)'.format(config.save_checkpoints_steps,
                                         config.save_checkpoints_secs))
    print('------------------------------------------')

    tfnet_est = TFNetEstimator(**net_config,
                               model_dir=FLAGS.model_dir,
                               add_summaries=_summaries,
                               optimizer=optimizers[FLAGS.optimizer],
                               weight_decay=FLAGS.weight_decay,
                               config=config)

    hooks = []
    if FLAGS.profile:
        hooks += [tf.train.ProfilerHook(output_dir=FLAGS.model_dir,
                                        save_steps=500,
                                        show_memory=False),
                 ]

    if FLAGS.enable_tracer:
        try:
            from tftracer import TracingServer
            tracing_server = TracingServer(server_port=8888)
            hooks += [tracing_server.hook]
        except ImportError:
            tf.logging.warn("tensorflow-tracer not available. Will not be "
                            "enabled")

    if FLAGS.testset:
        #eval_summary_hook = tf.train.SummarySaverHook(
        #    save_steps=1,
        #    summary_op=tf.summary.merge_all('audio_samples'))

        train_spec = tf.estimator.TrainSpec(train_input_fn, hooks=hooks)
        eval_spec = tf.estimator.EvalSpec(eval_input_fn,
                                          steps=None,
                                          #hooks=[eval_summary_hook]
                                         )
        while True:
            try:
                tf.estimator.train_and_evaluate(estimator=tfnet_est,
                                                train_spec=train_spec,
                                                eval_spec=eval_spec)
            except tf.estimator.NanLossDuringTrainingError:
                tf.logging.warn("NaN loss encountered. Attempting to continue")
                continue
            break
    else:
        tfnet_est.train(input_fn=train_input_fn,
                        hooks=hooks
                       )


if __name__ == '__main__':
    tf.app.run(main)
