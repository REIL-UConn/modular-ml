from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import jax
    import tensorflow as tf
    import torch

    TorchTensor = torch.Tensor
    TorchModule = torch.nn.Module

    JaxArray = jax.Array
    TensorFlowTensor = tf.Tensor
else:
    TorchTensor = Any
    TorchModule = Any

    JaxArray = Any
    TensorFlowTensor = Any
