"""Typical audio signal processing functions"""
import numpy as np
from scipy.signal import decimate, resample
from scipy import interpolate

def downsample(x, rate):
    x_ds = decimate(x, rate, axis=0, zero_phase=True)
    return x_ds

def upsample(x, rate):
    x_us = resample(x, len(x)*rate, axis=0)
    return x_us

def upsample_spline(x, rate):
    """Upsample using b-spline"""
    channels = x.shape[1]
    x_us = np.stack([_spline_us(x[:, i], rate) for i in range(channels)],
                    axis=1)

    return x_us

def _spline_us(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)

    x_sp = interpolate.splev(i_hr, f)

    return x_sp

def not_silent(x, threshold=0.05):
    return np.sum(np.square(x)) > threshold

def sample_segments(x, seg_length=8192,
                    stride=None,
                    keep_ratio=0.5,
                    filter_silence=True,
                    silence_threshold=0.05
                   ):
    """Returns Iterable
    x should be a single audio example with 1 or 2 channels
      first dimension is time, 2nd dimension is channel
    """
    assert len(x.shape) == 2

    n = x.shape[0]

    stride = int(seg_length/16) if stride is None else stride

    x_segs = [x[i:i+seg_length] for i in range(0, n-seg_length, stride)]

    x_segs = filter(lambda x: (np.random.uniform() <= keep_ratio), x_segs)

    if filter_silence:
        x_segs = filter(lambda x: not_silent(x, silence_threshold),
                        x_segs)

    return x_segs
