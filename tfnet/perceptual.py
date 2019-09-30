"""Common signal processing functions used in perceptual audio coding"""
import numpy as np
import tensorflow as tf
from . import ops

def sine_window(window_length, dtype=tf.float32):
    """Returns a sine window of length `window_length`, type `dtype`"""
    with tf.name_scope('sine_window'):
        n = tf.cast(tf.range(window_length), dtype)
        N = tf.cast(window_length, dtype)
        window_weights = tf.math.sin(np.pi * (n + 0.5) / N)
    return window_weights

def _get_transform(transform):
    if callable(transform):
        return transform
    if transform == 'stft' or transform == None:
        return tf.signal.rfft
    raise ValueError(f"{transform} is not a supported transform")

def _get_inverse_transform(transform):
    if callable(transform):
        return transform
    if transform == 'stft' or transform == None:
        return tf.signal.irfft
    raise ValueError(f"{transform} is not a supported transform")

def analysis(x,
             frame_length=512,
             frame_step=None,
             pad_signal=True,
             window_fn=sine_window,
             transform=tf.signal.rfft,
             **kwargs
            ):
    """Expects x to be [?, t, c]
    returns x spectrogram in [?, t, f, c]
    """
    transform = _get_transform(transform)
    if frame_step is None:
        frame_step = frame_length//2
    if pad_signal:
        x_padded = tf.pad(x, [[0, 0], [frame_step, frame_step], [0, 0]])
    x_padded = tf.transpose(x_padded, [0, 2, 1])
    x_frames = tf.signal.frame(x_padded,
                               frame_length=frame_length,
                               frame_step=frame_step,
                              )

    if window_fn is not None:
        window = window_fn(frame_length, dtype=x_frames.dtype)
        x_frames *= window

    x_spectrum = transform(x_frames,
                           **kwargs)
    x_spectrum = tf.transpose(x_spectrum, [0, 2, 3, 1])
    return x_spectrum

def analysis_w_mag_arg(x, **kwargs):
    """Convenience function for using with old code base,
    wrapper on analysis that returns X, |X|, arg(X)
    """
    X = analysis(x, **kwargs)
    if X.dtype is tf.complex64 or X.dtype is tf.complex128:
        Xmag = ops.mag(X)
        Xarg = ops.arg(X)
    else:
        Xmag = ops.mag(X)
        Xarg = None
    return X, Xmag, Xarg

def synthesis(x_spectrum,
              frame_length=512,
              frame_step=None,
              window_fn=sine_window,
              transform=tf.signal.irfft,
              **kwargs):
    """Inverse of `analysis`."""

    transform = _get_inverse_transform(transform)

    if frame_step is None:
        frame_step = frame_length//2

    frames = int(x_spectrum.shape[1])
    channels = int(x_spectrum.shape[-1])
    x_spectrum = tf.transpose(x_spectrum, [0, 3, 1, 2])
    y_frames = transform(x_spectrum,
                         **kwargs
                        )
    if window_fn is not None:
        window = window_fn(frame_length, x_spectrum.dtype.real_dtype)
        y_frames *= window

    y = tf.signal.overlap_and_add(y_frames, frame_step)

    y = y[:, :, frame_step:frame_step*frames]
    y = tf.transpose(y, [0, 2, 1])
    y.set_shape([None, frame_step*(frames-1), channels])
    return y

