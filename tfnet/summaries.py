"""Summaries types to log into tensorboard"""
import tensorflow as tf

def audio_sample_summary(sample_rate, max_outputs=8):
    return lambda name, pred, gt: tf.summary.audio(name,
                                                   pred if pred is not None else gt,
                                                   sample_rate=sample_rate,
                                                   max_outputs=max_outputs,
                                                   collections=[tf.GraphKeys.SUMMARIES,
                                                                'audio_samples']
                                                  )

def get_spectrogram(x, frame_size=512, frame_step=256):
    X = tf.signal.stft(x,
                       frame_length=frame_size,
                       frame_step=frame_step,
                      )
    X_mag = tf.abs(X)
    return tf.log(X_mag)[:, :, :, tf.newaxis]

def per_channel_spectrum(name, x, max_outputs=8):
    _, _, cc = x.get_shape()
    for c in range(cc):
        tf.summary.image('{}_{}'.format(name, c),
                         get_spectrogram(x[:, :, c]),
                         collections=[tf.GraphKeys.SUMMARIES,
                                      'audio_samples'],
                         max_outputs=max_outputs
                        )

def audio_spectrogram_summary(max_outputs=8):
    return lambda name, pred, gt: per_channel_spectrum(name,
                                                       pred if pred is not None else gt,
                                                       max_outputs=max_outputs
                                                      )
