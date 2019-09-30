"""Network implementation for TF-Net UNets
"""
import tensorflow as tf
from . import ops
from . import utils
from . import perceptual
import tensorflow.keras.layers as layers
from tensorflow.keras import activations
from functools import partial

LOG_D = tf.logging.debug
LOG_I = tf.logging.info

#slim = tf.contrib.slim #pylint: disable=invalid-name

NFILTERS = (128, 256, 512, 512)
FILTER_SIZES = (65, 33, 17, 9)

LOSS_TYPES = ['l2', 'snr', 'lsd' ]

_DOWNSAMPLE_DEFAULT = {'kernel_initializer':tf.orthogonal_initializer,
                       'activation':ops.lrelu,
                       'padding':'SAME',
                       'use_bias':False,
                       'strides':2}

_BOTTLENECK_DEFAULT = {'filters':NFILTERS[-1],
                       'kernel_size':FILTER_SIZES[-1],
                       'activation':ops.lrelu,
                       'padding':'SAME',
                       'kernel_initializer':tf.orthogonal_initializer,
                       'strides':2,
                      }

_UPSAMPLE_DEFAULT = {'kernel_initializer':tf.orthogonal_initializer,
                     'activation':tf.nn.relu,
                     'use_bias':False,
                     'padding':'SAME'}

def downsample(x, is_training, #pylint: disable=unused-argument
               reuse=False,    #pylint: disable=unused-argument
               nfilters=NFILTERS, filtersizes=FILTER_SIZES,
               out_collection='ds',
               ds_params=None
              ):
    """Downsample part of Audio UNet"""
    ds_params = ds_params if ds_params else _DOWNSAMPLE_DEFAULT
    net = x
    with tf.variable_scope(out_collection):
    #with slim.arg_scope([slim.conv1d],
    #                    outputs_collections=out_collection,
    #                    scope=out_collection,
    #                    **ds_params):
        net = layers.Conv1D(filters=nfilters[0],
                            kernel_size=filtersizes[0],
                            **ds_params)(net)
        tf.add_to_collection(out_collection, net)

        #net = slim.conv1d(x,
        #                       nfilters[0], filtersizes[0],
        #                       normalizer_fn=None)
        conv_params = list(zip(nfilters[1:], filtersizes[1:]))
        for nf, ks in conv_params:
            net = layers.Conv1D(filters=nf,
                                kernel_size=ks,
                                **ds_params)(net)
            tf.add_to_collection(out_collection, net)
        #net = slim.stack(net, slim.conv1d, conv_params)
    return net

def bottleneck_layer(x, reuse=tf.AUTO_REUSE, is_training=True,
                     params=None, name='bottleneck'):
    """Bottleneck layer in UNet"""
    params = params if params else _BOTTLENECK_DEFAULT
    with tf.variable_scope(name, reuse=reuse):
        net = layers.Conv1D(**params)(x)
        #net = slim.conv1d(x, **params)
        net = layers.Dropout(rate=0.5)(net, training=is_training)
    return net

#for compatibility
#bottleneck = bottleneck_layer

def upsample(net, is_training, #pylint: disable=unused-argument
             reuse=False,      #pylint: disable=unused-argument
             nfilters=NFILTERS, filtersizes=FILTER_SIZES,
             concat_collection='ds',
             us_params=None,
             out_collection='us'):
    """
    Upsample half of the UNet architecture
    """
    us_params = us_params if us_params else _UPSAMPLE_DEFAULT

    ds_ep = utils.convert_collection_to_dict(concat_collection)

    for k, val in ds_ep.items():
        LOG_D((k, val))

    #pylint: disable=invalid-name
    for ii, (nf, fs, l_ep) in enumerate(reversed(list(zip(nfilters,
                                                          filtersizes,
                                                          ds_ep)))):
    #pylint: enable=invalid-name
        with tf.variable_scope('us_conv{}'.format(ii)):
            LOG_D('------')
            LOG_D(net)
            #net = slim.conv1d(net, 2*nf, fs)
            net = layers.Conv1D(2*nf, fs, **us_params)(net)
            tf.add_to_collection(out_collection, net)
            LOG_D(net)
            net = layers.Dropout(rate=0.5)(net, training=is_training)
            #net = slim.dropout(net, 0.5,
            #                   is_training=is_training)
            LOG_D(net)
            net = ops.subpixel1d(net, r=2)
            LOG_D((net, ds_ep[l_ep]))
            net = layers.Concatenate(-1)([net, ds_ep[l_ep]])

    return net

def finalconv(net, is_training=None, reuse=False, params=None):
    """
    Final conv layer after upsample portion of unets
    """
    #Unsed, to keep function signature same
    _ = is_training
    _ = params
    with tf.variable_scope('finalconv', reuse=reuse):
        net = layers.Conv1D(2, 9,
                            activation=None,
                            padding='SAME')(net)
        net = ops.subpixel1d(net, r=2)

    return net

def unet(x, is_training,
         reuse=tf.AUTO_REUSE,
         nfilters=NFILTERS, filtersizes=FILTER_SIZES,
         ds_params=None,
         us_params=None,
         bottleneck_params=None,
         netname='unet'
        ):
    """docstring for audiounet"""

    ds_params = ds_params if ds_params else _DOWNSAMPLE_DEFAULT
    us_params = us_params if us_params else _UPSAMPLE_DEFAULT
    bottleneck_params = bottleneck_params if ds_params else _BOTTLENECK_DEFAULT

    LOG_D('{} filters: {}'.format(netname, nfilters))
    LOG_D('{} filtersizes: {}'.format(netname, filtersizes))
    ds_collection = '_'.join(filter(None, [tf.get_variable_scope().name,
                                           netname,
                                           'ds']))

    with tf.variable_scope(netname, reuse=reuse):
        net = downsample(x,
                         is_training,
                         reuse,
                         nfilters, filtersizes,
                         ds_params=ds_params,
                         out_collection=ds_collection)
        net = bottleneck_layer(net, reuse=reuse,
                               is_training=is_training,
                               params=bottleneck_params)
        #tf.add_to_collection('unet_latent', net)
        net = upsample(net,
                       is_training,
                       reuse,
                       nfilters, filtersizes,
                       us_params=us_params,
                       concat_collection=ds_collection,
                       out_collection='us')
        net = finalconv(net)

    return net

def frequency_time_layer(x, filters,
                         frequency_kernel_size,
                         time_kernel_size=1,
                         strides=1,
                         frequency_activation_fn=None,
                         time_activation_fn=None,
                         transposed=False,
                         **kwargs
                        ):
    """Conv over frequency dim only, activate, followed by time only,
    then activation. Time dimension does not change shape
    """
    Conv_f = layers.Conv2DTranspose if transposed else layers.Conv2D
    activation_f = activations.get(frequency_activation_fn)

    activation_t = activations.get(time_activation_fn)

    x = Conv_f(filters=filters,
               kernel_size=(1, frequency_kernel_size),
               strides=(1, strides),
               padding='same',
               **kwargs
              )(x)
    x = activation_f(x)

    if time_kernel_size>1:
        x = layers.Conv2D(filters=filters,
                          kernel_size=(time_kernel_size, 1),
                          padding='same',
                          **kwargs)(x)
        x = activation_t(x)

    return x

def sbr_branch(x, is_training,
               reuse=tf.AUTO_REUSE,
               netname='spectral',
               rate=2,
               frame_length=512,
               transform='stft',
               **kwargs):

    with tf.variable_scope(netname, reuse=reuse):
        X = perceptual.analysis(x,
                                frame_length=frame_length,
                                transform=transform,
                               )
        Xmag = tf.abs(X)
        if Xmag.get_shape()[-2] % 2 == 1:
            X_dc, X_f = ops.spectral_copies(Xmag, rate, expand=False)
        else:
            X_dc = None
            X_f = Xmag

        in_channels = X.get_shape().as_list()[-1]

        filters = [in_channels, 32, 64, 128, 256]
        ds = [0]
        for ff in filters[1:]:
            X_f = frequency_time_layer(X_f,
                                       filters=ff,
                                       frequency_kernel_size=5,
                                       frequency_activation_fn=ops.lrelu,
                                       strides=2,
                                      )
            ds.append(X_f)

        ww = int(X_f.get_shape()[2])

        tf.add_to_collection('unet_latent', X_f)

        X_f = frequency_time_layer(X_f,
                                   filters=512,
                                   frequency_kernel_size=ww,
                                   frequency_activation_fn=ops.lrelu,
                                   strides=2,
                                   time_kernel_size=5,
                                   time_activation_fn=ops.lrelu,
                                  )
        filters.reverse()
        ds.reverse()
        for ff, dd in zip(filters, ds):
            X_f = frequency_time_layer(X_f,
                                       filters=ff,
                                       frequency_kernel_size=5,
                                       frequency_activation_fn=ops.lrelu,
                                       strides=2,
                                       transposed=True,
                                      )
            X_f = X_f + dd
        if X_dc is not None:
            NET = tf.concat([X_dc, X_f], axis=-2)
        else:
            NET = X_f

    return NET, X

def spectral_transform(x):
    """rfft on x
    Variable naming conventions:
        CAPs are frequency domain, smalls are time domain
    """
    #pylint: disable=invalid-name
    with tf.name_scope('spectral_transform'):
        X = ops.rfft(x)
        Xmag = tf.abs(X)
        Xarg = ops.arg(X)
    #pylint: enable=invalid-name
    return X, Xmag, Xarg

def inv_spectral_transform(X):
    y = ops.irfft(X)
    return y

def audiounet_spectral(x, is_training,
                       reuse=tf.AUTO_REUSE,
                       nfilters=NFILTERS, filtersizes=FILTER_SIZES,
                       netname='spectral',
                       rate=1,
                       **kwargs
                      ):
    """Spectral version of AudioUNet"""
    #small letters for time, CAPs for frequency
    #pylint: disable=invalid-name

    with tf.variable_scope(netname, reuse=reuse):
        X, Xmag, _ = spectral_transform(x)
        if Xmag.get_shape()[-2]%2 == 1:
            X_dc, X_f = ops.spectral_copies(Xmag, rate)
        else:
            X_dc = None
            X_f =  Xmag
        X_f = unet(X_f, is_training, reuse,
                   nfilters=nfilters, filtersizes=filtersizes,
                   **kwargs
                  )
        if X_dc is not None:
            NET = tf.concat([X_dc, X_f], axis=1)
        else:
            NET = X_f
    #pylint: enable=invalid-name

    return NET, X

def _timefreq_weighted_average(x1, x2, clamp):
    weight_var = clamp(tf.get_variable('fusion_weights_unclamped', x1.get_shape()[1:]))
    y = x1 * (1-weight_var) + x2 * weight_var
    return y, weight_var

def _timefreq_predict_weights_fdense_0(x1, x2, clamp):
    """x1, x2 shape  [n, t, f, c]
    Weighted average of X1 and X2 using weights predicted by a small network
    that uses x1 and x2 as input
    """
    _w = layers.Concatenate(axis=-1)([x1, x2])
    #_w is now (n, t, f, 2c), f*2c not expected to be very big
    #       (512 frequencies bins for stft/mdct of length 1024 max)
    _t, _f, _c = _w.get_shape().as_list()[-3:]
    _w = layers.Conv2D(kernel_size=(1, _f), filters=(_f*_c))(_w)
    #_w is now (n, t, 1, 2*f*c)
    _w = layers.Conv2D(kernel_size=(5, 1), filters=(_f), padding='same')(_w)
    _w = layers.Conv2D(kernel_size=(5, 1), filters=(_f), padding='same')(_w)
    _w = layers.Activation('sigmoid')(_w)
    #_w is now (n, t, 1, f)
    _w = tf.reshape(_w, [-1, _t, _f, 1])
    y = x1 * (1-_w) + x2 * _w

    return y, _w

def _timefreq_predict_weights_from_var(x1, x2, clamp, in_var):
    """Fuse x1 and x2 using weights predicted from variables given in in_var"""
    print("from_var")
    _t, _f, _c = x1.get_shape().as_list()[-3:]
    _w = tf.get_collection(in_var)
    if len(_w) >1:
        _w = layers.Concatenate(axis=-1)(_w)
    else:
        _w = _w[0]
    _wt, _wf, _wc = _w.get_shape().as_list()[-3:]
    _w = layers.Conv2D(kernel_size=(7, 1), filters=32, padding='same')(_w)
    _w = layers.Conv2D(kernel_size=(1, _wf), filters=_f, padding='valid')(_w)
    _w = layers.Activation('sigmoid')(_w)
    _w = tf.transpose(_w, [0, 1, 3, 2])
    y = x1 * (1-_w) + x2 * _w

    return y, _w

def _get_fusion_op(op_fn_or_name):
    print("Getting fusion op {}".format(op_fn_or_name))
    if callable(op_fn_or_name):
        return op_fn_or_name
    if op_fn_or_name == 'ave':
        return _timefreq_weighted_average
    if op_fn_or_name == 'fdense_0':
        return _timefreq_predict_weights_fdense_0
    if op_fn_or_name == 'from_unet':
        return partial(_timefreq_predict_weights_from_var, in_var='unet_latent')
    if op_fn_or_name == 'time':
        return lambda x, y: x, 0
    if op_fn_or_name == 'freq':
        return lambda x, y: y, 0
    raise ValueError("{} is not a supported fusion method".format(op_fn_or_name))

def fusion(time_net, spectral_net, clamp=lambda x: (tf.tanh(x)+1)/2,
           analysis_transform=spectral_transform,
           synthesis_transform=inv_spectral_transform,
           fusion_op=_timefreq_weighted_average,
           is_training=True):
    """Fusion layer to combine predictions from time and frequency branches"""

    #small letters for time, CAPs for frequency
    #pylint: disable=invalid-name
    x1 = time_net
    X2 = spectral_net

    _ = is_training

    with tf.variable_scope('fusion'):
        if X2 is None:
            g = x1
            fusion_weights = 0
        else:
            #This can be improved
            X1, X1mag, X1arg = analysis_transform(x1)

            if X1mag is not None:
                Gmag, fusion_weights = fusion_op(X1mag, X2, clamp)

                G = ops.complexmagarg(Gmag, X1arg)
            else:
                G, fusion_weights = fusion_op(X1, X2, clamp)
            g = synthesis_transform(G)

    #pylint: enable=invalid-name
    return g, fusion_weights

def default_net(loss='l2', copies_rate=1):
    """Network with all default parameters"""
    netdef = {'time_fn': lambda x, is_training: unet(x, is_training=is_training),
              'freq_fn': lambda x, is_training: audiounet_spectral(x, is_training=is_training,
                                                                   rate=copies_rate,
                                                                  )[0],
              'fusion_fn':
              lambda time_branch, freq_branch, is_training: fusion(time_branch,
                                                                   freq_branch,
                                                                   is_training=is_training)[0],
              'loss_fn': loss
             }

    return netdef

def build_net(loss='l2', copies_rate=1, time_params=None, freq_params=None,
              window_length=0, transform='stft', fusion_op='ave',
             ):
    if time_params is None and freq_params is None:
        return default_net(loss, copies_rate)

    if window_length == 0:
        netdef = {'time_fn': lambda x, is_training: unet(x, is_training=is_training,
                                                         **time_params,
                                                        ),
                  'freq_fn': lambda x, is_training: audiounet_spectral(x, is_training=is_training,
                                                                       rate=copies_rate,
                                                                       **freq_params,
                                                                      )[0],
                  'fusion_fn':
                  lambda time_branch, freq_branch, is_training: fusion(time_branch,
                                                                       freq_branch,
                                                                       is_training=is_training)[0],
                  'loss_fn': loss
                 }
    else:
        analysis = partial(perceptual.analysis_w_mag_arg,
                           frame_length=window_length,
                           transform=transform)
        synthesis = partial(perceptual.synthesis, transform=transform,
                            frame_length=window_length,
                           )
        netdef = {'time_fn': lambda x, is_training: unet(x, is_training=is_training,
                                                         **time_params,
                                                        ),
                  'freq_fn': lambda x, is_training: sbr_branch(x, is_training=is_training,
                                                               rate=copies_rate,
                                                               frame_length=window_length,
                                                               transform=transform
                                                              )[0],
                  'fusion_fn':
                  lambda time_branch, freq_branch, is_training: fusion(time_branch,
                                                                       freq_branch,
                                                                       analysis_transform=analysis,
                                                                       synthesis_transform=synthesis,
                                                                       fusion_op=_get_fusion_op(fusion_op),
                                                                       is_training=is_training)[0],
                  'loss_fn': loss
                 }

    if fusion_op == 'time':
        netdef['freq_fn'] = lambda x, is_training: None
    if fusion_op == 'freq':
        netdef['time_fn'] = lambda x, is_training: None

    return netdef
