import numpy as np
import torch


def xavier_init(size):
    """Xavier initialization function.

    Args:
      - size: input data dimension (tuple or list)
    """
    in_dim = size[0]
    xavier_stddev = (2.0 / in_dim) ** 0.5  # Calculate stddev based on input dimension
    return torch.randn(*size) * xavier_stddev  # Generate tensor with specified stddev


def batch_generator(x, t, y, batch_size):
    """Generate mini-batch with x, t, and y.

    Args:
      - x: features (torch.Tensor)
      - t: treatments (torch.Tensor)
      - y: observed labels (torch.Tensor)
      - batch_size: mini batch size

    Returns:
      - X_mb: mini-batch features (torch.Tensor)
      - T_mb: mini-batch treatments (torch.Tensor)
      - Y_mb: mini-batch observed labels (torch.Tensor)
    """
    num_samples = x.size(0)
    # Generate random indices for the mini-batch
    indices = torch.randint(0, num_samples, (batch_size,))
    X_mb = x[indices]
    # Reshape treatments and labels to [batch_size, 1]
    T_mb = t[indices].view(batch_size, 1)
    Y_mb = y[indices].view(batch_size, 1)
    return X_mb, T_mb, Y_mb
