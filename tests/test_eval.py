"""Tests for eval mode"""
import unittest
import sys
import os
import tensorflow as tf
from tfnet import TFNetEstimator
from tfnet import nets
import datahelper.dataset as ds

from tests.constants import * #pylint: disable=wildcard-import,unused-wildcard-import

class TestTFNetEval(unittest.TestCase):

    @unittest.skipIf(not os.path.isdir(DUMMY_MODEL_PATH),
                     """Dummy model to load does not exist,
                     create with tests/create_dummy_model.py""")
    def test_loadmodel(self):
        """Test running eval from with trained model"""
        tf.logging.set_verbosity(tf.logging.INFO)

        path = PATH.decode(sys.stdout.encoding)
        #2 files, 64 epochs, batchsize 32 => 2*64/32 = 4 iterations
        dset = ds.dataset_with_preprocess(LISTFILE_1, path,
                                          epochs=1,
                                          batchsize=16,
                                         )
        #RunConfig for more more printing since we are only training for very few steps
        config = tf.estimator.RunConfig(log_step_count_steps=1)

        tfnet_est = TFNetEstimator(**nets.default_net(), config=config,
                                   model_dir=DUMMY_MODEL_PATH
                                  )

        tfnet_est.evaluate(
            input_fn=lambda: dset.make_one_shot_iterator().get_next())

        self.assertIsNotNone(tfnet_est)
