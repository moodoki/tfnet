"""Command line args for train.py"""
import tensorflow as tf
from tfnet import nets
from . import distribute
from . import branch_args
from .branch_args import get_time_params, get_freq_params

tf.app.flags.DEFINE_string('trainset', 'trainset.txt',
                           """Text file of filenames of files to be used as
                           training set"""
                          )
tf.app.flags.DEFINE_string('testset', None,
                           """Text file of filenames of files to be used as
                           evaluation set"""
                          )
tf.app.flags.DEFINE_string('datapath', 'data',
                           """Path to be prepended to the filenames in
                           trainset"""
                          )
tf.app.flags.DEFINE_integer('audio_sample_rate', -1,
                            """Sampling rate of audio files. Used for
                            generating tensorboard summaries. set to 0 for to
                            automatically determine base on input, or set to
                            any negative value to disable generating audio
                            summaries""")
tf.app.flags.DEFINE_enum('objective', 'l2', nets.LOSS_TYPES,
                         """Which objective to optimize""")
tf.app.flags.DEFINE_integer('downsample_rate', 2,
                            """Downsample input audio by this rate to generate
                            training input""")
tf.app.flags.DEFINE_bool('spectral_copies', False,
                         """Apply spectral copies on frequency branch""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                          """Learning rate""")
tf.app.flags.DEFINE_bool('learning_rate_decay', False,
                         """Use polynomial learning rate decay""")
tf.app.flags.DEFINE_float('weight_decay', -1,
                          """Weight decay regularization weight. negative
                          values disables weight decay""")
tf.app.flags.DEFINE_integer('window_length', 0,
                            """Use windowed version of network with given window
                            length, if window_length==0, non-windowed version is
                            used.""")
tf.app.flags.DEFINE_enum('transform', 'stft', ['stft'],
                         """Transform to use if window_length is specified.
                         Can be mdct or stft. Defaults to stft, has no effect
                         if window lenght is 0""")
tf.app.flags.DEFINE_enum('fusion_op', 'ave',
                         ['ave', 'fdense_0', 'from_unet', 'time', 'freq'],
                         """Fusion op to use, defaults to use a learnt average
                         weight""")

tf.app.flags.DEFINE_enum('optimizer', 'sgd', ['sgd', 'adam'],
                         """Which optimizer to use""")
tf.app.flags.DEFINE_string('model_dir', 'sandpit/model',
                           """Path to where checkpoints, logs and models are to
                           be saved. If directory exists, training will
                           continue from where it was last saved.  """
                          )
tf.app.flags.DEFINE_integer('epochs', 10,
                            """Number of epochs to train for"""
                           )
tf.app.flags.DEFINE_integer('batchsize', 16,
                            """Batchsize"""
                           )
tf.app.flags.DEFINE_bool('multigpu', False,
                         """Use multiple GPUs on local machine using
                         MirroredStrategy""")
tf.app.flags.DEFINE_integer('log_step_count_steps', 100,
                            """Prints some text summary every this number of
                            steps""")
tf.app.flags.DEFINE_bool('debug', False,
                         """Debug mode, prints a lot more stuff""")

tf.app.flags.DEFINE_bool('profile', False,
                         """Generate a timeline-xx.json to for profiling.
                         Useful in finding out where the bottleneck is. One
                         will be generated every 500 steps""")
tf.app.flags.DEFINE_bool('enable_tracer', False,
                         """Enable tensorflow-tracer for performance tracing""")
tf.app.flags.DEFINE_bool('usexla', False,
                         """Use experimental XLA JIT-compilation""")
tf.app.flags.DEFINE_integer('save_checkpoints_steps', None,
                            """Frequency of checkpointing, in number of steps""")
tf.app.flags.DEFINE_integer('save_checkpoints_secs', None,
                            """Frequency of checkpointing (in time).
                            If time is used, steps should not be set""")

FLAGS = tf.app.flags.FLAGS
