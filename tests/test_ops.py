"""Unit tests for operations defined in ops.py against reference numpy
implementation"""
import unittest
import numpy as np
import tensorflow as tf

import tfnet.ops as ops

class TestOps(unittest.TestCase):

    def test_lsd(self):
        """Check LSD tf op against numpy reference implementation"""
        x = np.random.rand(6, 10, 1).astype(np.float32)
        y = np.random.rand(6, 10, 1).astype(np.float32)
        x_f = np.fft.fft(x, axis=1)
        y_f = np.fft.fft(y, axis=1)

        true_X = np.log10(np.abs(x_f)**2)
        reco_X = np.log10(np.abs(y_f)**2)
        reco_X_diff_squared = (true_X - reco_X)**2
        reco_lsd = np.mean(np.sqrt(np.mean(reco_X_diff_squared, axis=1)))

        tflsd_op = ops.lsd_loss(tf.constant(x), tf.constant(y))

        with tf.Session() as sess:
            tflsd = sess.run(tflsd_op)

        print("Numpy:", reco_lsd)
        print("TF: ", tflsd)
        print(reco_lsd/tflsd)

        self.assertTrue(np.allclose(reco_lsd, tflsd))

    def test_fft(self):
        """FFT should be equivalent to numpy fft on axis 1"""
        x = np.random.rand(6, 10, 1).astype(np.float32)
        x_f = np.fft.fft(x, axis=1)

        x_f_tfop = ops.fft(tf.constant(x))
        with tf.Session() as sess:
            x_f_tf = sess.run(x_f_tfop)

        self.assertTrue(np.allclose(x_f, x_f_tf, atol=1e-6))

    def test_spectral_copies_degenrate(self):
        """ degerate copies should just split into dc and non-dc components"""
        a = np.array([np.array(range(17))[:, np.newaxis] for _ in range(4)])
        a_gt = np.array([np.array(range(1,17))[:, np.newaxis]
                         for _ in range(4)])
        a_dc = np.array([[[0]] for _ in range(4)])

        at = tf.constant(a)
        copies_op = ops.spectral_copies(at, 1)
        with tf.Session() as sess:
            tf_dc, tf_copies = sess.run(copies_op)

        self.assertTrue(np.array_equal(tf_dc, a_dc))
        self.assertTrue(np.array_equal(tf_copies, a_gt))


    def test_spectral_copies(self):
        """ copies should copy dim2 by n times """
        a = np.array([np.array(range(17))[:, np.newaxis] for _ in range(4)])
        a_gt = np.array([np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])[:, np.newaxis]
                         for _ in range(4)])
        a_dc = np.array([[[0]] for _ in range(4)])

        at = tf.constant(a)
        copies_op = ops.spectral_copies(at, 4)
        with tf.Session() as sess:
            tf_dc, tf_copies = sess.run(copies_op)

        self.assertTrue(np.array_equal(tf_dc, a_dc))
        self.assertTrue(np.array_equal(tf_copies, a_gt))

    def test_spectral_copies_2d(self):
        #(4, 10, 5, 1) array with 0, 1, 2, 3, 4 in dim 3
        a = np.array([[np.array(range(5))[:, np.newaxis] for _ in range(10)] for _ in range(4)])
        a_gt = np.array([[np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])[:, np.newaxis]
                          for _ in range(10)] for _ in range(4)])
        a_dc = np.array([[[[0]] for _ in range(10)] for _ in range(4)])

        at = tf.constant(a)
        copies_op = ops.spectral_copies(at, 4)
        with tf.Session() as sess:
            tf_dc, tf_copies = sess.run(copies_op)

        self.assertTrue(np.array_equal(tf_dc, a_dc))
        self.assertTrue(np.array_equal(tf_copies, a_gt))

    def test_spectral_unroll(self):
        """ unroll should copy and reverse even copies  dim2 by n times """
        a = np.array([np.array(range(17))[:, np.newaxis] for _ in range(4)])
        a_gt = np.array([np.array([1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4, 4, 3, 2, 1])[:, np.newaxis]
                         for _ in range(4)])
        a_dc = np.array([[[0]] for _ in range(4)])

        at = tf.constant(a)
        copies_op = ops.spectral_unroll(at, 4)
        with tf.Session() as sess:
            tf_dc, tf_copies = sess.run(copies_op)

        self.assertTrue(np.array_equal(tf_dc, a_dc))
        self.assertTrue(np.array_equal(tf_copies, a_gt))

