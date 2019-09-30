"""Dummy dataset for unittests"""
import unittest
import numpy as np
import tensorflow as tf

def get_dummy_dataset(length=8192, channels=1, count=16,
                      batchsize=4, repeat=200,
                      drop_remainder=True
                     ):
    """Dummy dataset generator for use in unit tests"""
    dummy_hr = np.array(np.linspace(0, 1, length)[:, np.newaxis], dtype=np.float32)
    dummy_hr = np.hstack([dummy_hr for _ in range(channels)])
    dummy_lr = dummy_hr.copy()
    dummy_lr[1::2] = 0

    dummy_train = [(dummy_lr.copy(), dummy_hr.copy()) for _ in range(count)]

    dummy_dset = tf.data.Dataset.from_generator(lambda: ((l, h) for l, h in dummy_train),
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=([length, channels],
                                                               [length, channels]))

    #16 samples per epoch, 2 epochs, batch size 4 -> 8 iterations
    dummy_dset = dummy_dset.repeat(repeat).batch(batchsize, drop_remainder=drop_remainder)
    return dummy_dset

class TestDummyDataset(unittest.TestCase):

    def test_shapecheck(self):
        """Basic smoke tests for dummy dataset used in other unit tests,
        Actual values don't really matter, just shapes count
        """
        #sample len 10, 2 channles, batch size 4
        dset = get_dummy_dataset(10, 2, batchsize=4).make_one_shot_iterator().get_next()
        self.assertEqual(dset[0].shape.as_list(), [4, 10, 2])
        self.assertEqual(dset[1].shape.as_list(), [4, 10, 2])

    def test_shapecheck_8192_1_32(self):
        dset = get_dummy_dataset(8192, 1, batchsize=32).make_one_shot_iterator().get_next()
        self.assertEqual(dset[0].shape.as_list(), [32, 8192, 1])
        self.assertEqual(dset[1].shape.as_list(), [32, 8192, 1])

    def test_shapecheck_nodrop(self):
        """No dropping of samples, batch size should be ?"""
        #sample len 4, 2 channles, batch size 4
        dset = get_dummy_dataset(10, 2, batchsize=4,
                                 drop_remainder=False).make_one_shot_iterator().get_next()
        self.assertEqual(dset[0].shape.as_list(), [None, 10, 2])
        self.assertEqual(dset[1].shape.as_list(), [None, 10, 2])

    def test_shapecheck_nodrop_8192_1_32(self):
        dset = get_dummy_dataset(8192, 1, batchsize=32,
                                 drop_remainder=False).make_one_shot_iterator().get_next()
        self.assertEqual(dset[0].shape.as_list(), [None, 8192, 1])
        self.assertEqual(dset[1].shape.as_list(), [None, 8192, 1])
