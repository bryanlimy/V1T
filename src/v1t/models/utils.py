import torch
import collections
import numpy as np
import typing as t
from torch import nn
from math import floor


def int2tuple(value: t.Union[int, t.Tuple[int, int]]):
    return (value, value) if isinstance(value, int) else value


def conv2d_shape(
    input_shape: t.Tuple[int, int, int],
    num_filters: int,
    kernel_size: t.Union[int, t.Tuple[int, int]],
    stride: t.Union[int, t.Tuple[int, int]] = 1,
    padding: t.Union[int, t.Tuple[int, int]] = 0,
    dilation: t.Union[int, t.Tuple[int, int]] = 1,
):
    """Calculate 2D convolution output shape given input shape in shape (C,H,W)"""
    kernel_size = int2tuple(kernel_size)
    stride = int2tuple(stride)
    padding = int2tuple(padding)
    dilation = int2tuple(dilation)
    new_h = (
        (input_shape[1] + (2 * padding[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1)
        / stride[0]
    ) + 1

    new_w = (
        (input_shape[2] + (2 * padding[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1)
        / stride[1]
    ) + 1

    return (num_filters, floor(new_h), floor(new_w))


def pool2d_shape(
    input_shape: t.Tuple[int, int, int],
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1,
):
    if stride is None:
        stride = kernel_size

    new_h = (
        (input_shape[1] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride
    ) + 1
    new_w = (
        (input_shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride
    ) + 1
    return (input_shape[0], floor(new_h), floor(new_w))


def transpose_conv2d_shape(
    input_shape: t.Tuple[int, int, int],
    num_filters: int,
    kernel_size: t.Union[int, t.Tuple[int, int]],
    stride: t.Union[int, t.Tuple[int, int]] = 1,
    padding: t.Union[int, t.Tuple[int, int]] = 0,
    output_padding: t.Union[int, t.Tuple[int, int]] = 0,
    dilation: t.Union[int, t.Tuple[int, int]] = 1,
):
    kernel_size = int2tuple(kernel_size)
    stride = int2tuple(stride)
    padding = int2tuple(padding)
    output_padding = int2tuple(output_padding)
    dilation = int2tuple(dilation)
    new_h = (
        (input_shape[1] - 1) * stride[0]
        - 2 * padding[0]
        + dilation[0] * (kernel_size[0] - 1)
        + output_padding[0]
        + 1
    )
    new_w = (
        (input_shape[2] - 1) * stride[1]
        - 2 * padding[1]
        + dilation[1] * (kernel_size[1] - 1)
        + output_padding[1]
        + 1
    )
    return (num_filters, floor(new_h), floor(new_w))


class ELU1(nn.Module):
    """ELU + 1 activation to output standardized responses"""

    def __init__(self):
        super(ELU1, self).__init__()
        self.elu = nn.ELU()
        self.register_buffer("one", torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor):
        return self.elu(inputs) + self.one


class DropPath(nn.Module):
    """
    Stochastic depth for regularization https://arxiv.org/abs/1603.09382
    Reference:
    - https://github.com/aanna0701/SPT_LSA_ViT/blob/main/utils/drop_path.py
    - https://github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dropout: float = 0.0):
        super(DropPath, self).__init__()
        assert 0 <= dropout <= 1
        self.register_buffer("keep_prop", torch.tensor(1 - dropout))

    def forward(self, inputs: torch.Tensor):
        if self.keep_prop == 1 or not self.training:
            return inputs
        shape = (inputs.size(0),) + (1,) * (inputs.ndim - 1)
        random_tensor = torch.rand(shape, dtype=inputs.dtype, device=inputs.device)
        random_tensor = torch.floor(self.keep_prop + random_tensor)
        outputs = (inputs / self.keep_prop) * random_tensor
        return outputs


class BufferDict(nn.Module):
    """Holds buffers in a dictionary.

    Reference: https://botorch.org/api/utils.html#botorch.utils.torch.BufferDict

    BufferDict can be indexed like a regular Python dictionary, but buffers it
    contains are properly registered, and will be visible by all Module methods.

    :class:`~torch.nn.BufferDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.BufferDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~torch.nn.BufferDict` (the argument to
      :meth:`~torch.nn.BufferDict.update`).

    Note that :meth:`~torch.nn.BufferDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Args:
        buffers (iterable, optional): a mapping (dictionary) of
            (string : :class:`~torch.Tensor`) or an iterable of key-value pairs
            of type (string, :class:`~torch.Tensor`)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.buffers = nn.BufferDict({
                        'left': torch.randn(5, 10),
                        'right': torch.randn(5, 10)
                })

            def forward(self, x, choice):
                x = self.buffers[choice].mm(x)
                return x
    """

    def __init__(self, buffers=None):
        r"""
        Args:
            buffers: A mapping (dictionary) from string to :class:`~torch.Tensor`, or
                an iterable of key-value pairs of type (string, :class:`~torch.Tensor`).
        """
        super(BufferDict, self).__init__()
        if buffers is not None:
            self.update(buffers)

    def __getitem__(self, key):
        return self._buffers[key]

    def __setitem__(self, key, buffer):
        self.register_buffer(key, buffer)

    def __delitem__(self, key):
        del self._buffers[key]

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.keys())

    def __contains__(self, key):
        return key in self._buffers

    def clear(self):
        """Remove all items from the BufferDict."""
        self._buffers.clear()

    def pop(self, key):
        r"""Remove key from the BufferDict and return its buffer.

        Args:
            key (string): key to pop from the BufferDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self):
        r"""Return an iterable of the BufferDict keys."""
        return self._buffers.keys()

    def items(self):
        r"""Return an iterable of the BufferDict key/value pairs."""
        return self._buffers.items()

    def values(self):
        r"""Return an iterable of the BufferDict values."""
        return self._buffers.values()

    def update(self, buffers):
        r"""Update the :class:`~torch.nn.BufferDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        .. note::
            If :attr:`buffers` is an ``OrderedDict``, a :class:`~torch.nn.BufferDict`,
            or an iterable of key-value pairs, the order of new elements in it is
            preserved.

        Args:
            buffers (iterable): a mapping (dictionary) from string to
                :class:`~torch.Tensor`, or an iterable of
                key-value pairs of type (string, :class:`~torch.Tensor`)
        """
        if not isinstance(buffers, collections.abc.Iterable):
            raise TypeError(
                "BuffersDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(buffers).__name__
            )

        if isinstance(buffers, collections.abc.Mapping):
            if isinstance(buffers, (collections.OrderedDict, BufferDict)):
                for key, buffer in buffers.items():
                    self[key] = buffer
            else:
                for key, buffer in sorted(buffers.items()):
                    self[key] = buffer
        else:
            for j, p in enumerate(buffers):
                if not isinstance(p, collections.abc.Iterable):
                    raise TypeError(
                        "BufferDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(p).__name__
                    )
                if not len(p) == 2:
                    raise ValueError(
                        "BufferDict update sequence element "
                        "#" + str(j) + " has length " + str(len(p)) + "; 2 is required"
                    )
                self[p[0]] = p[1]

    def extra_repr(self):
        child_lines = []
        for k, p in self._buffers.items():
            size_str = "x".join(str(size) for size in p.size())
            device_str = "" if not p.is_cuda else " (GPU {})".format(p.get_device())
            parastr = "Buffer containing: [{} of size {}{}]".format(
                torch.typename(p), size_str, device_str
            )
            child_lines.append("  (" + k + "): " + parastr)
        tmpstr = "\n".join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError("BufferDict should not be called.")
