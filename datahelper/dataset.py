"""Data loading pipelines"""
import os
from functools import partial
import tensorflow as tf

import numpy as np
from scipy.io import wavfile

try:
    import librosa
except ImportError:
    tf.logging.warn("librosa is not available, loading from wav files might not work")

from . import filters

INFO = tf.logging.info
DEBUG = tf.logging.debug
ERROR = tf.logging.error

def load_fileslist(filename, path=None):
    """Reads a text file containing filenames into a list.
    Prepends optional path.

    returns tensorflow Dataset of filenames
    """
    with open(filename) as f:
        if path is not None:
            audio_filelist = [os.path.join(path, line.strip()) for line in f if line.strip()]
        else:
            audio_filelist = [line.strip() for line in f if line.strip()]

    assert audio_filelist

    def _gen():
        for item in audio_filelist:
            yield item

    fileslist_dataset = tf.data.Dataset.from_generator(_gen,
                                                       output_types=tf.string
                                                      )

    return fileslist_dataset

def _audio_to_float(data):
    """Some audio files are not of float type. This ensures that the data is
    in the 0-1 range and of type float32"""
    if data.dtype == np.float32:
        return data
    return np.true_divide(data, np.iinfo(data.dtype).max, dtype=np.float32)

def _load_wav(filename,
              trim_silence=None,
              gt_rate=16000,
             ):
    """Loads wav file given by the filename using scipy,
    datatype is as-is in the wav file. Needs to be normalized to float32"""
    data, _ = librosa.load(filename, sr=gt_rate)
    if trim_silence:
        data, _ = librosa.effects.trim(data, top_db=trim_silence)
    data = _audio_to_float(data)
    if len(data.shape) == 1:
        data = data[:, np.newaxis]
    return data

def audio_dataset_from_fileslist(filename, path=None,
                                 num_parallel_calls=4,
                                 trim_silence=None,
                                 gt_rate=16000,
                                ):
    fn_dset = load_fileslist(filename, path)
    read_wav = lambda x: tf.py_func(partial(_load_wav,
                                            trim_silence=trim_silence,
                                            gt_rate=gt_rate,
                                           ),
                                    [x], tf.float32)
    audio_dset = fn_dset.map(read_wav, num_parallel_calls=num_parallel_calls)
    return audio_dset

def get_segment_dataset(dataset, length=8192, channels=1,
                        num_parallel_calls=4,
                        segs_per_sample=1,
                       ):
    """Get a random subsegment from dataset"""
    def _crop(data):
        return tf.random_crop(data, [length, channels])

    def _crop_multi(data):
        segs = []
        for _ in range(segs_per_sample):
            segs.append(tf.random_crop(data, [length, channels]))
        return tf.stack(segs)

    #if segs_per_sample == 1:
    #    return dataset.map(_crop, num_parallel_calls=num_parallel_calls)

    return dataset.filter(lambda x: tf.shape(x)[0] > length)\
            .map(_crop_multi, num_parallel_calls=num_parallel_calls)\
            .apply(tf.data.experimental.unbatch())


def downsample_by(x, rate=2):
    y = filters.downsample(x, rate)
    y = filters.upsample(y, rate)
    return y

def get_lq_hq_pair(dataset, degrade_fn, num_parallel_calls=4):
    """Takes a dataset with single (hq) audio example per row, returns
    a dataset that's a lq, hq pair generated using degrade_fn"""
    _degrade_fn = lambda x: tf.py_func(degrade_fn, [x], tf.float32)
    def _degrade(x):
        y = _degrade_fn(x)
        y.set_shape(x.shape)
        return y, x
    return dataset.map(_degrade, num_parallel_calls)

def dataset_with_preprocess(filelist, path=None,
                            degrade_fn=lambda x: downsample_by(x, 2),
                            epochs=1, batchsize=32,
                            drop_remainder=False,
                            length=8192, channels=1,
                            num_parallel_calls=4,
                            segs_per_sample=1,
                            shuffle=True,
                            **kwargs
                           ):
    """Full on-the-fly processing of high quality audio files to get training
    set"""
    DEBUG("Ignored args: " + str(kwargs))

    dset = audio_dataset_from_fileslist(filelist, path, num_parallel_calls)
    dset = dset.repeat(epochs)
    dset = get_segment_dataset(dset, length, channels, num_parallel_calls, segs_per_sample)
    dset = get_lq_hq_pair(dset, degrade_fn)
    if shuffle:
        dset = dset.shuffle(batchsize*4).batch(batchsize, drop_remainder)
    else:
        dset = dset.batch(batchsize, drop_remainder)
    return dset

def single_file_dataset(filename, upsample_rate=2, seg_length=8192, batchsize=16, **kwargs):
    """Loads a single audio file and process it in order, use for prediction"""
    DEBUG("Ignored args: " + str(kwargs))
    audio_in = filters.upsample(_load_wav(filename), upsample_rate)
    audio_len, channels = audio_in.shape
    padlen = seg_length - audio_len%seg_length
    audio_padded = np.pad(audio_in, [(0, padlen), (0, 0)], 'constant')
    audio_segs = audio_padded.reshape((-1, seg_length, channels))

    def _gen():
        for seg in audio_segs:
            yield seg

    dset = tf.data.Dataset.from_generator(_gen,
                                          output_types=tf.float32,
                                          output_shapes=[seg_length, channels])
    dset = dset.batch(batchsize)
    return dset

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_example(lq, hq):
    x_shape = lq.shape
    y_shape = hq.shape
    feature = {
        'data_shape': _int_feature(x_shape),
        'label_shape': _int_feature(y_shape),
        'data': _float_feature(lq.ravel()),
        'label': _float_feature(hq.ravel()),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(lq, hq):
    tf_string = tf.py_func(
        serialize_example,
        [lq, hq],
        tf.string,
    )
    return tf.reshape(tf_string, ())

def get_tfrecord_dataset(filename, batchsize=16, drop_remainder=False,
                         epochs=1,
                         shuffle=True,
                         upsample_fn=filters.upsample,
                         **kwargs):
    """Parsing for Prepared TFRecord dataset"""
    DEBUG("Ignored args: " + str(kwargs))
    #get dataset properties
    itr = tf.python_io.tf_record_iterator(filename)
    r = next(itr)
    example = tf.train.Example()
    example.ParseFromString(r)
    x_shape = example.features.feature['data_shape'].int64_list.value[:]
    y_shape = example.features.feature['label_shape'].int64_list.value[:]
    is_v2 = len(example.features.feature['label'].float_list.value[:]) > 0
    itr.close()
    print('---------------------------------------------------------------')
    print('TFRecord Dataset:', filename)
    print('data_shape:', x_shape, 'groundtruth_shape', y_shape)
    if is_v2:
        print('V2 dataset detected')
    else:
        print('V1 dataset detected')
    print('---------------------------------------------------------------')

    _noop = lambda x: x
    if x_shape[0] != y_shape[0]:
        _rate = y_shape[0] // x_shape[0]
        print(f"Input needs to be upsampled {_rate} times. "
              "Will be upsampled with a low-pass filter")
        lq_preop = lambda x: tf.py_func(lambda x: upsample_fn(x, _rate), [x], tf.float32)
    else:
        lq_preop = _noop

    def _parse_fn_v1(example):
        fmt = {
            'data': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            #'data_shape': tf.FixedLenFeature([], tf.int64),
            #'label_shape': tf.FixedLenFeature([], tf.int64)
        }

        parsed = tf.parse_single_example(example, fmt)

        lq_audio = tf.decode_raw(parsed['data'], tf.float32)
        lq_audio.set_shape(np.prod(x_shape))
        lq_audio = tf.reshape(lq_audio, x_shape)
        lq_audio = lq_preop(lq_audio)
        #py_func doesn't have shape hint, lq and hq should have same shape now
        lq_audio = tf.reshape(lq_audio, y_shape)

        hq_audio = tf.decode_raw(parsed['label'], tf.float32)
        hq_audio.set_shape(np.prod(y_shape))
        hq_audio = tf.reshape(hq_audio, y_shape)

        return lq_audio, hq_audio

    def _parse_fn_v2(example):
        fmt = {
            'data': tf.FixedLenFeature(x_shape, tf.float32),
            'label': tf.FixedLenFeature(y_shape, tf.float32),
        }
        parsed = tf.parse_single_example(example, fmt)
        return parsed['data'], parsed['label']

    _parse_fn = _parse_fn_v2 if is_v2 else _parse_fn_v1

    if shuffle:
        dset = lambda: tf.data.TFRecordDataset(filename)\
                .shuffle(batchsize*4)\
                .repeat(epochs)\
                .map(_parse_fn)\
                .batch(batchsize, drop_remainder)
    else:
        dset = lambda: tf.data.TFRecordDataset(filename)\
                .repeat(epochs)\
                .map(_parse_fn)\
                .batch(batchsize, drop_remainder)

    return dset

def get_dataset(filename, **params):
    """Picks the right dataset pipeline depending on filename"""
    if filename.lower().endswith('txt'):
        return lambda: dataset_with_preprocess(filename, **params)
    if filename.lower().endswith('wav'):
        return lambda: single_file_dataset(filename, **params)
    if filename.lower().endswith('tfrecord'):
        return get_tfrecord_dataset(filename, **params)

    raise TypeError('Unknown dataset type:' + str(filename))
