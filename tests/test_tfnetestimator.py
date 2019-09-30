"""
basic tests for TFNetEstimator
"""
import unittest
import tensorflow as tf

from tfnet import TFNetEstimator

from tests.dummydata import get_dummy_dataset

class TestTFNetEstimator(unittest.TestCase):
    def setUp(self):
        self.dummy_input = tf.placeholder(tf.float32, [None, 8192, 1])
        self.in_fn = lambda x, _: tf.layers.conv1d(x, 1, 3, padding='same')
        self.fusion_fn = lambda x, y, _: x+y
        tf.logging.set_verbosity(tf.logging.INFO)

    def test_smoketest_defaults(self):
        self.assertRaises(ValueError, TFNetEstimator)

    def test_smoketest_both(self):
        tfnet_est = TFNetEstimator(time_fn=self.in_fn, freq_fn=self.in_fn,
                                   fusion_fn=self.fusion_fn,
                                  )
        self.assertIsNotNone(tfnet_est)

    def test_smoketest_time_only(self):
        tfnet_est = TFNetEstimator(time_fn=self.in_fn,
                                   fusion_fn=self.fusion_fn,
                                  )
        self.assertIsNotNone(tfnet_est)

    def test_smoketest_freq_only(self):
        tfnet_est = TFNetEstimator(freq_fn=self.in_fn,
                                   fusion_fn=self.fusion_fn,
                                  )
        self.assertIsNotNone(tfnet_est)

    def test_tfnetest_train(self):
        """test training with dummy data"""

        tfnet_est = TFNetEstimator(time_fn=self.in_fn, freq_fn=self.in_fn,
                                   fusion_fn=self.fusion_fn,
                                   loss_fn=tf.losses.mean_squared_error
                                  )
        self.assertIsNotNone(tfnet_est.train(
            input_fn=lambda: get_dummy_dataset().make_one_shot_iterator().get_next()
        ))

    def test_tfnetest_eval(self):
        """test training with dummy data"""

        tfnet_est = TFNetEstimator(time_fn=self.in_fn, freq_fn=self.in_fn,
                                   fusion_fn=self.fusion_fn,
                                   loss_fn=tf.losses.mean_squared_error
                                  )
        self.assertIsNotNone(tfnet_est.evaluate(
            input_fn=lambda: get_dummy_dataset().make_one_shot_iterator().get_next()
        ))

    def test_tfnetest_pred(self):
        """test training with dummy data"""

        tfnet_est = TFNetEstimator(time_fn=self.in_fn, freq_fn=self.in_fn,
                                   fusion_fn=self.fusion_fn,
                                   loss_fn=tf.losses.mean_squared_error
                                  )
        self.assertIsNotNone(tfnet_est.predict(
            input_fn=lambda: get_dummy_dataset().make_one_shot_iterator().get_next()
        ))
