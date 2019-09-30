"""Tests for predict mode"""
import unittest
import sys
import os
import tensorflow as tf
from tfnet import TFNetEstimator
from tfnet import nets
import datahelper.dataset as ds

from tests.constants import * #pylint: disable=wildcard-import,unused-wildcard-import

class TestTFNetPred(unittest.TestCase):

    @unittest.skipIf(not os.path.isdir(DUMMY_MODEL_PATH),
                     """Dummy model to load does not exist,
                     create with tests/create_dummy_model.py""")
    def test_loadmodel(self):
        """Test running eval from with trained model"""
        tf.logging.set_verbosity(tf.logging.INFO)

        path = PATH.decode(sys.stdout.encoding)
        #2 files, 64 epochs, batchsize 32 => 2*64/32 = 4 iterations
        dset = ds.single_file_dataset(LQ_AUDIO_FILE,
                                     )
        #RunConfig for more more printing since we are only training for very few steps
        config = tf.estimator.RunConfig(log_step_count_steps=1)

        tfnet_est = TFNetEstimator(**nets.default_net(), config=config,
                                   model_dir=DUMMY_MODEL_PATH
                                  )

        preds = tfnet_est.predict(
            input_fn=lambda: dset.make_one_shot_iterator().get_next())

        for pred in preds:
            self.assertEqual(pred.shape, (8192, 1))
