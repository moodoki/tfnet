"""Unittests for datahelper"""
import unittest
import sys

import tensorflow as tf
import datahelper as dh

from tests.constants import * #pylint: disable=wildcard-import,unused-wildcard-import
#pylint: disable=missing-docstring

class TestFilenamesFromTxt(unittest.TestCase):
    def setUp(self):
        dset = dh.dataset.load_fileslist(LISTFILE_1)
        self.itr = dset.make_one_shot_iterator().get_next()

    def test_readsfile(self):
        with tf.Session() as sess:
            for item in LISTFILE_1_CONTENTS:
                self.assertEqual(item, sess.run(self.itr))

class TestFilenamesFromTxtPath(unittest.TestCase):
    def setUp(self):
        self.path = PATH.decode(sys.stdout.encoding)

    def test_readsfile(self):
        dset = dh.dataset.load_fileslist(LISTFILE_1, self.path)
        itr = dset.make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            for item in LISTFILE_1_CONTENTS:
                self.assertEqual(PATH+b'/'+item, sess.run(itr))

    def test_readwav(self):
        """Only shapes are checked"""
        dset = dh.dataset.audio_dataset_from_fileslist(LISTFILE_1, self.path)
        itr = dset.make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            for item in AUDIO_SHAPES:
                audio = sess.run(itr)
                self.assertEqual(audio.shape, item)

    def test_random_crop_single(self):
        dset = dh.dataset.audio_dataset_from_fileslist(LISTFILE_1, self.path)
        dset = dh.dataset.get_segment_dataset(dset)
        itr = dset.make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            audio_seg = sess.run(itr)
            self.assertEqual(audio_seg.shape, (8192, 1))
