import tensorflow as tf
from tensorflow.python.client import device_lib

def get_local_gpus():
    """Returns a list of names of local GPUS"""
    devices = device_lib.list_local_devices()
    gpu_names = [dev.name for dev in devices if dev.device_type == 'GPU']
    return gpu_names

def multi_gpu_config(**kwargs):
    """Generate RunConfig for Multi-GPU training with Mirrored Strategy"""
    local_gpus = get_local_gpus()
    if not local_gpus:
        tf.logging.warn("No GPUs found! Falling back to defaults")
        return tf.estimator.RunConfig(**kwargs)
    mirrored_strat = tf.distribute.MirroredStrategy(get_local_gpus())
    return tf.estimator.RunConfig(train_distribute=mirrored_strat,
                                  **kwargs
                                 )
