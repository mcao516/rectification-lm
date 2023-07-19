import torch
import numpy as np
import torch.nn.functional as F


def sequence_mask(lengths, max_len=None, dtype=None, device=None) :
    r"""Return a mask tensor representing the first N positions of each cell.
    If ``lengths`` has shape ``[d_1, d_2, ..., d_n]`` the resulting tensor
    ``mask`` has dtype ``dtype`` and shape ``[d_1, d_2, ..., d_n, maxlen]``,
    with
    ```
    mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
    ```
    Examples:
    ```python
    sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                 #  [True,  True,  True, False, False],
                                 #  [True,  True, False, False, False]]
    sequence_mask([[1, 3],[2,0]])  # [[[ True, False, False],
                                   #   [ True,  True,  True]],
                                   #  [[ True,  True, False],
                                   #   [False, False, False]]]
    ```
    Args:
        lengths: integer tensor or list of int, all its values <= max_len.
        max_len: scalar integer tensor, size of last dimension of returned
            tensor. Default is the maximum value in ``lengths``.
        dtype: the desired data type of returned tensor. Default: if None,
            returns :torch:`ByteTensor`.
        device: the desired device of returned tensor. Default: if None, uses
            the current device for the default tensor type.
    Returns:
        A mask tensor of shape :python:`lengths.shape + (max_len,)`, cast to
        specified dtype.
    Raises:
        ValueError: if ``max_len`` is not a scalar.
    """
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, device=device)
    elif device is None:
        device = lengths.device
    lengths: torch.LongTensor
    if max_len is None:
        max_len = torch.max(lengths).item()

    size = lengths.size()
    row_vector = torch.arange(max_len, device=device, dtype=lengths.dtype).view(
        *([1] * len(size)), -1).expand(*size, max_len)
    mask = (row_vector < lengths.unsqueeze(-1)).to(device=device)
    if dtype is not None:
        mask = mask.to(dtype=dtype)

    return mask


def masked_reverse_cumsum(X, lengths, dim):
    """
    Args:
        X (Tensor): [batch_size, max_tgt_len]
        lengths (Tensor): [batch_size]
        dim (int): -1
        gamma (float): the discount factor
    
    """
    masked_X = X * sequence_mask(lengths, max_len=X.shape[1])
    return (masked_X
            .flip(dims=[dim])
            .cumsum(dim=dim)
            .flip(dims=[dim]))


def discounted_future_sum(values, lengths, num_steps=None, gamma=1.0):
    """
    Args:
        values (Tensor): reward values with size [batch_size, max_tgt_len]
        lengths (Tensor): target sequence length with size [batch_size]
        num_steps (int): number of future steps to sum over.
        gamma (float): discount value.
    
    Return:
        output (Tensor): [batch_size, max_tgt_len]
    """
    assert values.dim() == 2
    
    batch_size, total_steps = values.shape
    values = values * sequence_mask(lengths, max_len=values.shape[1])

    num_steps = total_steps if num_steps is None else num_steps
    num_steps = min(num_steps, total_steps)
    
    padding = torch.zeros([batch_size, num_steps - 1]).to(values)
    padded_values = torch.cat([values, padding], 1)
    discount_filter = gamma ** torch.arange(num_steps).to(values).reshape(1, 1, -1)

    output = F.conv1d(padded_values.unsqueeze(-2), discount_filter).squeeze(1)
    return output


def polyak_update(model, tgt_model, target_lr):
    for param_, param in zip(tgt_model.parameters(), model.parameters()):
        param_.data.copy_((1 - target_lr) * param_ + target_lr * param)
