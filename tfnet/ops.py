"""
Tensor ops used by TFNet
"""
import tensorflow as tf
from . import perceptual as p

#Consider using @export decorators?
__all__ = ['snr',
           'fft',
           'rfft',
           'ifft',
           'irfft',
           'subpixel1d',
           'arg',
           'mag',
           'complexmagarg',
           'lrelu'
          ]

def snr(x, y):
    """Computes the SNR of given signal x, with reference signal y"""
    l2 = tf.reduce_mean((x-y)**2 + 1e-6, axis=[1, 2]) #pylint: disable=invalid-name
    l2_norm = tf.reduce_mean(y**2, axis=[1, 2])
    _snr = 10 * tf.log(l2_norm / l2 + 1e-8) / tf.log(10.)

    return _snr

def snr_loss(references, predictions):
    with tf.variable_scope('snr_loss'):
        _snr_loss = tf.reduce_mean(snr(predictions, references))
    tf.losses.add_loss(_snr_loss)
    return _snr_loss

def l2_loss(references, predictions):
    return tf.losses.mean_squared_error(labels=references, predictions=predictions)

def log10(x):
    return tf.log(x)/tf.log(tf.constant(10, dtype=x.dtype))

def lsd(x, y):
    """Compute Log-spectral distance between reference signal x and predicted
    signal y"""
    #pylint: disable=invalid-name
    X = fft(x)
    Y = fft(y)
    Px = log10(tf.abs(X)**2)
    Py = log10(tf.abs(Y)**2)
    _lsd = tf.sqrt(tf.reduce_mean((Px-Py)**2, axis=1))
    #pylint: enable=invalid-name
    return tf.reduce_mean(_lsd, axis=1) #collapse/average across channels

def lsd_loss(reference, predictions):
    with tf.variable_scope('lsd_loss'):
        _lsd_loss = tf.reduce_mean(lsd(reference, predictions))
    tf.losses.add_loss(_lsd_loss)
    return _lsd_loss

def fft(x):
    """Performs FFT along the axis of interest.
    Convenience wrapper around tf.spectral.fft when for time series with
    multiple channels, in the format (Example, samples, channel)"""

    if x.dtype is tf.float32:
        x = tf.cast(x, tf.complex64)
    elif x.dtype is tf.float64:
        x = tf.cast(x, tf.complex128)

    x = tf.transpose(x, [0, 2, 1])
    x = tf.fft(x)
    x = tf.transpose(x, [0, 2, 1])

    return x

def rfft(x):
    """Performs FFT along the axis of interest for Real only signals
    Convenience wrapper around tf.spectral.rfft when for time series with
    multiple channels, in the format (Example, samples, channel)"""
    x = tf.transpose(x, [0, 2, 1])
    x = tf.spectral.rfft(x)
    x = tf.transpose(x, [0, 2, 1])

    return x

def ifft(x):
    x = tf.transpose(x, [0, 2, 1])
    x = tf.ifft(x)
    x = tf.transpose(x, [0, 2, 1])
    return x

def irfft(x):
    x = tf.transpose(x, [0, 2, 1])
    x = tf.spectral.irfft(x)
    x = tf.transpose(x, [0, 2, 1])
    return x

def arg(x):
    x_r = tf.real(x)
    x_i = tf.imag(x)
    return tf.atan2(x_i, x_r)

def mag(x):
    return tf.abs(x)

def complexmagarg(_mag, _arg):
    r = tf.cos(_arg)*_mag
    i = tf.sin(_arg)*_mag
    return tf.complex(r, i)

def subpixel1d(x, r, name='subpixel1d'):
    """1d Subpixel transform"""
    with tf.variable_scope(name):
        x = tf.transpose(x, [2, 1, 0])
        x = tf.batch_to_space_nd(x, [r], [[0, 0]])
        x = tf.transpose(x, [2, 1, 0])
    return x

def lrelu(x, leak=0.2, name='lrelu'):
    """Leaky-ReLU"""
    with tf.variable_scope(name):
        f_1 = 0.5*(1+leak)
        f_2 = 0.5*(1-leak)
        return f_1*x +f_2*tf.abs(x)

#pylint: disable=invalid-name
def spectral_copies(Xmag, rate=1, expand=True):
    """Apply spectral copies on input Xmag for specified rate

    eg, rate=2
    [ 0, 1, 2, 3, 4, 5, 6, 7, 8] -> [0], [1,2,3,4,1,2,3,4]

    expects data to be in [N, f, c] or [N, t, f, c]

    for rank=4 (i.e spectrograms) number of frequency bins will be expanded if
    expand=True(default). This argument is ignored for rank=3

    rate=1  is degenerate and just splits into dc and non-dc components,
    name_scope is not applied so that it's visible in tensorboard that this
    operation isn't doing anything significant."""
    rank = len(Xmag.get_shape())
    l = int((int(Xmag.shape[-2])-1)/rate) # number of non DC component not zero after LPF
    if rank == 3:
        get_dc = lambda x: x[:, 0, tf.newaxis, :] #tf.newaxis to prevent automatic flatten
        get_passband = lambda x: x[:, 1:l+1, :]
    elif rank == 4:
        get_dc = lambda x: x[:, :, 0, tf.newaxis, :]
        if expand:
            get_passband = lambda x: x[:, :, 1:, :]
        else:
            get_passband = lambda x: x[:, :, 1:l+1, :]

    with tf.name_scope('spectral_copies' if rate > 1 else None):
        X_dc = get_dc(Xmag)
        pass_band = get_passband(Xmag)
        X_f = tf.concat([pass_band
                         for _ in range(rate)],
                        axis=-2)
    return X_dc, X_f
#pylint: enable=invalid-name

#pylint: disable=invalid-name
def spectral_unroll(Xmag, rate=1):
    """Apply spectral copies on input Xmag for specified rate

    eg, rate=2
    [ 0, 1, 2, 3, 4, 5, 6, 7, 8] -> [0], [1,2,3,4,4,3,2,1]

    eg, rate=4
    [ 0, 1, 2, 3, 4, 5, 6, 7, 8] -> [0], [1,2,2,1,1,2,2,1]

    rate=1  is defenerate and just splits into dc and non-dc components,
    name_scope is not applied so that it's visible in tensorboard that this
    operation isn't doing anything significant."""
    with tf.name_scope('spectral_copies' if rate > 1 else None):
        X_dc = Xmag[:, 0, tf.newaxis, :] #tf.newaxis to prevent automatic flatten
        l = int((int(Xmag.shape[1])-1)/rate) # number of non DC component not zero after LPF
        pass_band = Xmag[:, 1:l+1, :]
        X_f = tf.concat([pass_band if ii%2 == 0 else tf.reverse(pass_band, [1])
                         for ii in range(rate)],
                        axis=1)
        return X_dc, X_f
#pylint: enable=invalid-name
