"""Unittests for on the fly training data pipeline"""
import unittest
import sys
import tensorflow as tf


import datahelper.dataset as ds

from tests.constants import * #pylint: disable=wildcard-import,unused-wildcard-import

class TestRandomSegmentBatch(unittest.TestCase):

    def setUp(self):
        self.path = PATH.decode(sys.stdout.encoding)
        dset = ds.audio_dataset_from_fileslist(LISTFILE_1, self.path)
        self.dset = ds.get_segment_dataset(dset)

    def test_segmentbatch(self):
        itr = self.dset.make_one_shot_iterator().get_next()
        self.assertEqual(itr.shape, (8192, 1))

    def test_train_data_pipe_default(self):
        dset = ds.dataset_with_preprocess(LISTFILE_1, self.path)
        itr = dset.make_one_shot_iterator().get_next()
        self.assertTrue(itr[0].shape.is_compatible_with(tf.TensorShape((None, 8192, 1))))
        self.assertTrue(itr[1].shape.is_compatible_with(tf.TensorShape((None, 8192, 1))))
        #defaults with test data shoudl throw after 1 iter
        with tf.Session() as sess:
            sess.run(itr)
            self.assertRaises(tf.errors.OutOfRangeError, lambda: sess.run(itr))

    def test_train_data_pipe_options(self):
        dset = ds.dataset_with_preprocess(LISTFILE_1, self.path,
                                          epochs=10,
                                          length=4096,
                                          batchsize=8,
                                          drop_remainder=True,
                                         )
        itr = dset.make_one_shot_iterator().get_next()
        self.assertTrue(itr[0].shape.is_compatible_with(tf.TensorShape((8, 4096, 1))))
        self.assertTrue(itr[1].shape.is_compatible_with(tf.TensorShape((8, 4096, 1))))
        #2 files, 10 epochs => 20 samples
        #batch of 8 => 2 full iters, 3rd should throw
        with tf.Session() as sess:
            sess.run(itr)
            sess.run(itr)
            self.assertRaises(tf.errors.OutOfRangeError, lambda: sess.run(itr))

    def test_train_data_pipe_multiseg(self):
        dset = ds.dataset_with_preprocess(LISTFILE_1, self.path,
                                          epochs=1,
                                          length=4096,
                                          batchsize=8,
                                          drop_remainder=True,
                                          segs_per_sample=20
                                         )
        itr = dset.make_one_shot_iterator().get_next()
        self.assertTrue(itr[0].shape.is_compatible_with(tf.TensorShape((8, 4096, 1))))
        self.assertTrue(itr[1].shape.is_compatible_with(tf.TensorShape((8, 4096, 1))))
        #2 files, 1 epochs, 20 segments per file => 40 samples
        #batch of 8 => 5 full iters, 6th should throw
        with tf.Session() as sess:
            sess.run(itr)
            sess.run(itr)
            sess.run(itr)
            sess.run(itr)
            sess.run(itr)
            self.assertRaises(tf.errors.OutOfRangeError, lambda: sess.run(itr))

      
