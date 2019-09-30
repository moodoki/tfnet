"""
Reimplementation of some functions in contrib as they are debing deprcated by
Tensorflow 
"""
from collections import OrderedDict
from tensorflow.python.framework import ops

def convert_collection_to_dict(collection, clear_collection=False):
    """Returns an OrderedDict of Tensors with their aliases as keys.
    Args:
      collection: A collection.
      clear_collection: When True, it clears the collection after converting to
        OrderedDict.
    Returns:
      An OrderedDict of {alias: tensor}
    """
    output = OrderedDict((alias, tensor)
                         for tensor in ops.get_collection(collection)
                         for alias in get_tensor_aliases(tensor))
    if clear_collection:
        ops.get_default_graph().clear_collection(collection)
    return output

def get_tensor_aliases(tensor):
    """Get a list with the aliases of the input tensor.
    If the tensor does not have any alias, it would default to its its op.name or
    its name.
    Args:
      tensor: A `Tensor`.
    Returns:
      A list of strings with the aliases of the tensor.
    """
    if hasattr(tensor, 'aliases'):
        aliases = tensor.aliases
    else:
        if tensor.name[-2:] == ':0':
            # Use op.name for tensor ending in :0
            aliases = [tensor.op.name]
        else:
            aliases = [tensor.name]
    return aliases
