"""Trains an estimator with test data"""
import unittest
import sys
import tensorflow as tf
from tfnet import TFNetEstimator
from tfnet import nets
import datahelper.dataset as ds

from tests.constants import * #pylint: disable=wildcard-import,unused-wildcard-import

class TestTrain(unittest.TestCase):
    def setUp(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = 1
        self.config = tf.estimator.RunConfig(log_step_count_steps=1,
                                             session_config=sess_config
                                            )

    def test_train(self):
        """Runs a training pass with test data for 4 iterations
        iterations of batchsize 16"""
        tf.logging.set_verbosity(tf.logging.INFO)
        path = PATH.decode(sys.stdout.encoding)
        #2 files, 64 epochs, batchsize 32 => 2*64/32 = 4 iterations
        dset = lambda: ds.dataset_with_preprocess(LISTFILE_1, path,
                                          epochs=64,
                                          batchsize=32,
                                         )
        tfnet_est = TFNetEstimator(**nets.default_net(), config=self.config)

        tfnet_est.train(
            input_fn=lambda: dset().make_one_shot_iterator().get_next())

        self.assertIsNotNone(tfnet_est)


    def test_traineval(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        path = PATH.decode(sys.stdout.encoding)
        #2 files, 64 epochs, batchsize 32 => 2*64/32 = 4 iterations
        dset = lambda: ds.dataset_with_preprocess(LISTFILE_1, path,
                                          epochs=4,
                                          batchsize=32,
                                          segs_per_sample=16,
                                         )
        dset_eval = lambda: ds.dataset_with_preprocess(LISTFILE_1, path,
                                               epochs=1,
                                               batchsize=16,
                                               segs_per_sample=16,
                                               shuffle=False
                                              )
        config = self.config.replace(save_checkpoints_steps=2)

        tfnet_est = TFNetEstimator(**nets.default_net(), config=config)

        input_fn = lambda: dset().make_one_shot_iterator().get_next()
        eval_input_fn = lambda: dset_eval().make_one_shot_iterator().get_next()

        train_spec = tf.estimator.TrainSpec(input_fn)
        eval_spec = tf.estimator.EvalSpec(eval_input_fn)

        tf.estimator.train_and_evaluate(tfnet_est, train_spec, eval_spec)


        self.assertIsNotNone(tfnet_est)

    def test_train_weightdecay(self):
        """Runs a training pass with test data for 4 iterations
        iterations of batchsize 16"""
        tf.logging.set_verbosity(tf.logging.INFO)
        path = PATH.decode(sys.stdout.encoding)
        #2 files, 64 epochs, batchsize 32 => 2*64/32 = 4 iterations
        dset = lambda: ds.dataset_with_preprocess(LISTFILE_1, path,
                                          epochs=64,
                                          batchsize=16,
                                         )
        tfnet_est = TFNetEstimator(**nets.default_net(),
                                   weight_decay=0.1,
                                   config=self.config)

        tfnet_est.train(
            input_fn=lambda: dset().make_one_shot_iterator().get_next())

        self.assertIsNotNone(tfnet_est)

    def test_train_sbr_fusion(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        path = PATH.decode(sys.stdout.encoding)
        #2 files, 64 epochs, batchsize 32 => 2*64/32 = 4 iterations
        dset = lambda: ds.dataset_with_preprocess(LISTFILE_1, path,
                                          epochs=64,
                                          batchsize=16,
                                         )
        netcfg = nets.build_net(copies_rate=2,
                                time_params={}, 
                                window_length=512)
        tfnet_est = TFNetEstimator(**netcfg,
                                   weight_decay=0.1,
                                   config=self.config)

        tfnet_est.train(
            input_fn=lambda: dset().make_one_shot_iterator().get_next())

        self.assertIsNotNone(tfnet_est)




