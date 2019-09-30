#!env python3
"""Used to generate a test model to be used by eval and prediction testcases
Uses same code as test_train testcase"""
import sys
import tensorflow as tf
from tfnet import TFNetEstimator
from tfnet import nets
import datahelper.dataset as ds

from tests.constants import * #pylint: disable=wildcard-import,unused-wildcard-import

def main(_):
    path = PATH.decode(sys.stdout.encoding)
    dset = ds.dataset_with_preprocess(LISTFILE_1, path,
                                      epochs=1,
                                      batchsize=32,
                                      segs_per_sample=64,
                                     )
    tfnet_est = TFNetEstimator(**nets.default_net(),
                               model_dir='tests/dummymodel'
                              )

    tfnet_est.train(
        input_fn=lambda: dset.make_one_shot_iterator().get_next())


if __name__ == '__main__':
    tf.app.run(main)
