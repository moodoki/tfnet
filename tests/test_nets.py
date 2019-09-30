"""
Smoketests for nets implementation
tests creation of graphs with default sized (?, 8192, 1) inputs
"""
import unittest
import tensorflow as tf
import tfnet.nets as nets
from tfnet import perceptual
from tfnet import ops

class TestSmokeNets(unittest.TestCase):

    def test_audiounet(self):
        x = tf.placeholder(tf.float32, [None, 8192, 1])
        net = nets.unet(x, is_training=False)
        self.assertEqual(net.shape.as_list(), x.shape.as_list())

    def test_audiounet_spectral(self):
        """Checks shape of audio unet
        should be input shape // 2 + 1
        Also checks types
        1st returned is magniture, float32
        2nd returned is complex64
        """
        x = tf.placeholder(tf.float32, [None, 8192, 1])
        expected_shape = x.shape.as_list()
        expected_shape[1] = expected_shape[1]//2+1

        NET, X = nets.audiounet_spectral(x, is_training=False) #pylint: disable=invalid-name
        self.assertEqual(NET.shape.as_list(), expected_shape)
        self.assertEqual(NET.dtype, tf.float32)
        self.assertEqual(X.shape.as_list(), expected_shape)
        self.assertEqual(X.dtype, tf.complex64)

    def test_audiounet_both(self):
        """TF net feedforward defaults creation test"""
        x = tf.placeholder(tf.float32, [None, 8192, 1])

        net = nets.unet(x, is_training=False)
        NET, _ = nets.audiounet_spectral(x, is_training=False) #pylint: disable=invalid-name

        net, _ = nets.fusion(net, NET)
        self.assertEqual(net.shape.as_list(), x.shape.as_list())

    def test_sbr_branch_512(self):
        x = tf.placeholder(tf.float32, [None, 8192, 1])

        NET, X = nets.sbr_branch(x, is_training=False,
                                 frame_length=512,
                                )

        self.assertEqual(NET.get_shape().as_list(), [None, 33, 257, 1])
        self.assertEqual(X.get_shape().as_list(), [None, 33, 257, 1])

    def test_sbr_branch_256(self):
        x = tf.placeholder(tf.float32, [None, 256, 1])

        NET, X = nets.sbr_branch(x, is_training=False,
                                 frame_length=256,
                                )

        self.assertEqual(NET.get_shape().as_list(), [None, 3, 129, 1])
        self.assertEqual(X.get_shape().as_list(), [None, 3, 129, 1])

    def test_fusion_sbr(self):
        x = tf.placeholder(tf.float32, [None, 8192, 1])

        net = nets.unet(x, is_training=False)
        NET, _ = nets.sbr_branch(x, is_training=False)
        NET = tf.abs(NET)

        y, _ = nets.fusion(net, NET,
                           analysis_transform=perceptual.analysis_w_mag_arg,
                           synthesis_transform=perceptual.synthesis,
                           is_training=False
                          )

        self.assertEqual(y.get_shape().as_list(),
                         x.get_shape().as_list()
                        )

    def tearDown(self):
        tf.reset_default_graph()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    unittest.main()
