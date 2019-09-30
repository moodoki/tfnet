"""Construct branch configuration from command line args"""
import tensorflow as tf
from tfnet.nets import _BOTTLENECK_DEFAULT

def _to_int_list(str_list):
    return [int(n) for n in str_list]

def _is_int_list(ll):
    if not ll:
        return True
    try:
        _to_int_list(ll)
    except:
        return False
    return True


tf.app.flags.DEFINE_list("time_nfilters", None,
                         "comma seperated number of filters in time branch")
tf.app.flags.DEFINE_list("freq_nfilters", None,
                         "comma seperated number of filters in freq branch")
tf.app.flags.DEFINE_list("time_filtersize", None,
                         "comma seperated filter size in time branch")
tf.app.flags.DEFINE_list("freq_filtersize", None,
                         "comma seperated filter size in freq branch")

tf.app.flags.register_validator("time_nfilters", _is_int_list,
                                "List of comma seperated ints expected")
tf.app.flags.register_validator("freq_nfilters", _is_int_list,
                                "List of comma seperated ints expected")
tf.app.flags.register_validator("time_filtersize", _is_int_list, 
                                "List of comma seperated ints expected")
tf.app.flags.register_validator("freq_filtersize", _is_int_list, 
                                "List of comma seperated ints expected")

def get_time_params():
    FLAGS = tf.app.flags.FLAGS
    time_params = {}
    bottleneck_params = _BOTTLENECK_DEFAULT
    if FLAGS.time_nfilters:
        time_params['nfilters'] = _to_int_list(FLAGS.time_nfilters)
        bottleneck_params['filters'] = time_params['nfilters'][-1]
    if FLAGS.time_filtersize:
        time_params['filtersizes'] = _to_int_list(FLAGS.time_filtersize)
        bottleneck_params['kernel_size'] = time_params['filtersizes'][-1]
    time_params['bottleneck_params'] = bottleneck_params

    return time_params if time_params else None

def get_freq_params():
    FLAGS = tf.app.flags.FLAGS
    freq_params = {}
    bottleneck_params = _BOTTLENECK_DEFAULT
    if FLAGS.freq_nfilters:
        freq_params['nfilters'] = _to_int_list(FLAGS.freq_nfilters)
        bottleneck_params['filters'] = freq_params['nfilters'][-1]
    if FLAGS.freq_filtersize:
        freq_params['filtersizes'] = _to_int_list(FLAGS.freq_filtersize)
        bottleneck_params['kernel_size'] = freq_params['filtersizes'][-1]
    freq_params['bottleneck_params'] = bottleneck_params

    return freq_params if freq_params else None
