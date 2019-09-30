"""Preprocessing functions for audio implemented as tensor ops

"""
from scipy.signal import decimate, resample

def downsample(x, rate):
    x_ds = decimate(x, rate, axis=0, zero_phase=True)
    return x_ds.astype(x.dtype)

def upsample(x, rate):
    x_us = resample(x, len(x)*rate, axis=0)
    return x_us.astype(x.dtype)
