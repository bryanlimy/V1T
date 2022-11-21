from .core import Core, register

import torch
import warnings
import typing as t
import numpy as np
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import Iterable, OrderedDict


class AttentionConv(nn.Module):
    """
    Implementation adapted from https://github.com/leaderj1001/Stand-Alone-Self-Attention
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
    ):
        """
        Parameters are intended to behave equivalently (and therefore sever as a drop-in replacement) to `torch.Conv2d`.
        Nevertheless, the underlying mechanism is conceptually different.
        Refer to https://arxiv.org/pdf/1906.05909.pdf for more information.
        """
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert (
            self.out_channels % self.groups == 0
        ), "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(
            torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True
        )
        self.rel_w = nn.Parameter(
            torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True
        )

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(
            3, self.kernel_size, self.stride
        )
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(
            3, self.kernel_size, self.stride
        )

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(
            batch, self.groups, self.out_channels // self.groups, height, width, -1
        )
        v_out = v_out.contiguous().view(
            batch, self.groups, self.out_channels // self.groups, height, width, -1
        )

        q_out = q_out.view(
            batch, self.groups, self.out_channels // self.groups, height, width, 1
        )

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum("bnchwk,bnchwk -> bnchw", out, v_out).view(
            batch, -1, height, width
        )

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode="fan_out", nonlinearity="relu")
        init.kaiming_normal_(
            self.value_conv.weight, mode="fan_out", nonlinearity="relu"
        )
        init.kaiming_normal_(
            self.query_conv.weight, mode="fan_out", nonlinearity="relu"
        )

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

    @property
    def weight(self):
        return torch.cat(
            (self.key_conv.weight, self.value_conv.weight, self.query_conv.weight),
            dim=0,
        )


class AdaptiveELU(nn.Module):
    """
    ELU shifted by user specified values. This helps to ensure the output to stay positive.
    """

    def __init__(self, xshift: int, yshift: int, **kwargs):
        super(AdaptiveELU, self).__init__(**kwargs)

        self.x_shift = torch.tensor(xshift, dtype=torch.float)
        self.y_shift = torch.tensor(yshift, dtype=torch.float)
        self.elu = nn.ELU()

    def forward(self, inputs: torch.Tensor):
        return self.elu(inputs - self.x_shift) + self.y_shift


def laplace():
    """
    Returns a 3x3 laplace filter.

    """
    return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).astype(np.float32)[
        None, None, ...
    ]


def laplace5x5():
    """
    Returns a 5x5 LaplacianOfGaussians (LoG) filter.

    """
    return np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [1, 2, -16, 2, 1],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 0, 0],
        ]
    ).astype(np.float32)[None, None, ...]


def laplace7x7():
    """
    Returns a 7x7 LaplacianOfGaussians (LoG) filter.

    """
    return np.array(
        [
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 3, 3, 3, 1, 0],
            [1, 3, 0, -7, 0, 3, 1],
            [1, 3, -7, -24, -7, 3, 1],
            [1, 3, 0, -7, 0, 3, 1],
            [0, 1, 3, 3, 3, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
        ]
    ).astype(np.float32)[None, None, ...]


def laplace3d():
    l = np.zeros((3, 3, 3))
    l[1, 1, 1] = -6.0
    l[1, 1, 2] = 1.0
    l[1, 1, 0] = 1.0
    l[1, 0, 1] = 1.0
    l[1, 2, 1] = 1.0
    l[0, 1, 1] = 1.0
    l[2, 1, 1] = 1.0
    return l.astype(np.float32)[None, None, ...]


class Laplace(nn.Module):
    """
    Laplace filter for a stack of data. Utilized as the input weight regularizer.
    """

    def __init__(self, padding=None, filter_size=3):
        """
        Laplace filter for a stack of data.

        Args:
            padding (int): Controls the amount of zero-padding for the convolution operation
                            Default is half of the kernel size (recommended)

        Attributes:
            filter (2D Numpy array): 3x3 Laplace filter.
            padding_size (int): Number of zeros added to each side of the input image
                before convolution operation.
        """
        super().__init__()
        if filter_size == 3:
            kernel = laplace()
        elif filter_size == 5:
            kernel = laplace5x5()
        elif filter_size == 7:
            kernel = laplace7x7()

        self.register_buffer("filter", torch.from_numpy(kernel))
        self.padding_size = self.filter.shape[-1] // 2 if padding is None else padding

    def forward(self, x):
        return F.conv2d(x, self.filter, bias=None, padding=self.padding_size)


class LaplaceL2norm(nn.Module):
    """
    Normalized Laplace regularizer for a 2D convolutional layer.
        returns |laplace(filters)| / |filters|
    """

    def __init__(self, padding=None):
        super().__init__()
        self.laplace = Laplace(padding=padding)

    def forward(self, x, avg=False):
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2 = x.size()
        return agg_fn(self.laplace(x.view(oc * ic, 1, k1, k2)).pow(2)) / agg_fn(
            x.view(oc * ic, 1, k1, k2).pow(2)
        )


class DepthSeparableConv2d(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.add_module(
            "in_depth_conv", nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        )
        self.add_module(
            "spatial_conv",
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
                groups=out_channels,
            ),
        )
        self.add_module(
            "out_depth_conv", nn.Conv2d(out_channels, out_channels, 1, bias=bias)
        )


class Bias2DLayer(nn.Module):
    """
    Bias per channel for a given 2d input.
    """

    def __init__(self, channels, initial=0, **kwargs):
        """
        Args:
            channels (int): number of channels in the input.
            initial (int, optional): intial value. Defaults to 0.
        """
        super().__init__(**kwargs)

        self.bias = torch.nn.Parameter(torch.empty((1, channels, 1, 1)).fill_(initial))

    def forward(self, x):
        return x + self.bias


class Scale2DLayer(nn.Module):
    """
    Scale per channel for a given 2d input.
    """

    def __init__(self, channels, initial=1, **kwargs):
        """
        Args:
            channels (int): number of channels in the input.
            initial (int, optional): intial value. Defaults to 1.
        """
        super().__init__(**kwargs)

        self.scale = torch.nn.Parameter(torch.empty((1, channels, 1, 1)).fill_(initial))

    def forward(self, x):
        return x * self.scale


@register("stacked2d")
class Stacked2dCore(Core, nn.Module):
    """
    An instantiation of the Core base class. Made up of layers layers of nn.sequential modules.
    Allows for the flexible implementations of many different architectures, such as convolutional layers,
    or self-attention layers.
    """

    def __init__(
        self,
        args,
        input_shape: t.Tuple[int, int, int],
        hidden_channels: int = 64,
        input_kern: int = 9,
        hidden_kern: int = 7,
        skip: int = 0,
        stride: int = 1,
        final_nonlinearity: bool = True,
        elu_shift: t.Tuple[int, int] = (0, 0),
        bias: bool = True,
        momentum: float = 0.9,
        pad_input: bool = False,
        hidden_padding: t.Union[None, int, t.List[int]] = None,
        batch_norm: bool = True,
        batch_norm_scale: bool = True,
        final_batchnorm_scale: bool = True,
        independent_bn_bias: bool = True,
        hidden_dilation: int = 1,
        laplace_padding: t.Union[None, int] = None,
        stack: t.Union[None, int, Iterable] = -1,
        use_avg_reg: bool = False,
        depth_separable: bool = True,
        attention_conv: bool = False,
        linear: bool = False,
        nonlinearity_config=None,
        name: str = "Stacked2DCore",
    ):
        if depth_separable and attention_conv:
            raise ValueError("depth_separable and attention_conv can not both be true")

        self.batch_norm = batch_norm
        self.final_batchnorm_scale = final_batchnorm_scale
        self.bias = bias
        self.independent_bn_bias = independent_bn_bias
        self.batch_norm_scale = batch_norm_scale

        super(Stacked2dCore, self).__init__(
            args=args, input_shape=input_shape, name=name
        )
        regularizer_config = dict(padding=laplace_padding)
        self._input_weights_regularizer = LaplaceL2norm(**regularizer_config)
        self.num_layers = args.num_layers
        self.register_buffer("gamma_input", torch.tensor(args.core_reg_input))
        self.register_buffer("gamma_hidden", torch.tensor(args.core_reg_hidden))
        self.input_channels = self.input_shape[0]
        self.hidden_channels = hidden_channels
        self.skip = skip

        self.activation_fn = AdaptiveELU
        self.activation_config = (
            nonlinearity_config if nonlinearity_config is not None else {}
        )

        self.dropout_rate = args.dropout

        self.stride = stride
        self.use_avg_reg = use_avg_reg
        if use_avg_reg:
            warnings.warn(
                "The averaged value of regularizer will be used.", UserWarning
            )
        self.hidden_padding = hidden_padding
        self.input_kern = input_kern
        self.hidden_kern = hidden_kern
        self.laplace_padding = laplace_padding
        self.hidden_dilation = hidden_dilation
        self.final_nonlinearity = final_nonlinearity
        self.elu_xshift, self.elu_yshift = elu_shift
        self.momentum = momentum
        self.pad_input = pad_input
        if stack is None:
            self.stack = range(self.num_layers)
        else:
            self.stack = (
                [*range(self.num_layers)[stack:]] if isinstance(stack, int) else stack
            )
        self.linear = linear

        if depth_separable:
            self.conv_layer_name = "ds_conv"
            self.ConvLayer = DepthSeparableConv2d
            self.ignore_group_sparsity = True
        elif attention_conv:
            # TODO: check if name attention_conv is backwards compatible
            self.conv_layer_name = "attention_conv"
            self.ConvLayer = self.AttentionConvWrapper
            self.ignore_group_sparsity = True
        else:
            self.conv_layer_name = "conv"
            self.ConvLayer = nn.Conv2d
            self.ignore_group_sparsity = False

        if self.ignore_group_sparsity and self.gamma_hidden > 0:
            warnings.warn(
                "group sparsity can not be calculated for the requested conv "
                "type. Hidden channels will not be regularized and gamma_hidden "
                "is ignored."
            )
        self.set_batchnorm_type()
        self.features = nn.Sequential()
        self.add_first_layer()
        self.add_subsequent_layers()
        self.initialize()
        self.output_shape = (
            hidden_channels,
            input_shape[1] - input_kern + 1,
            input_shape[2] - input_kern + 1,
        )

    def set_batchnorm_type(self):
        self.batchnorm_layer_cls = nn.BatchNorm2d
        self.bias_layer_cls = Bias2DLayer
        self.scale_layer_cls = Scale2DLayer

    def penultimate_layer_built(self):
        """Returns True if the penultimate layer has been built."""
        return len(self.features) == self.num_layers - 1

    def add_bn_layer(self, layer, hidden_channels):
        if self.batch_norm:
            if self.independent_bn_bias:
                layer["norm"] = self.batchnorm_layer_cls(
                    hidden_channels, momentum=self.momentum
                )
            else:
                layer["norm"] = self.batchnorm_layer_cls(
                    hidden_channels,
                    momentum=self.momentum,
                    affine=self.bias
                    and self.batch_norm_scale
                    and (
                        not self.penultimate_layer_built() or self.final_batchnorm_scale
                    ),
                )
                if self.bias and (
                    not self.batch_norm_scale
                    or (
                        self.penultimate_layer_built()
                        and not self.final_batchnorm_scale
                    )
                ):
                    layer["bias"] = self.bias_layer_cls(hidden_channels)
                elif self.batch_norm_scale and not (
                    self.penultimate_layer_built() and not self.final_batchnorm_scale
                ):
                    layer["scale"] = self.scale_layer_cls(hidden_channels)

    def add_activation(self, layer):
        if self.linear:
            return
        if not self.penultimate_layer_built() or self.final_nonlinearity:
            if self.activation_fn == AdaptiveELU:
                layer["nonlin"] = AdaptiveELU(self.elu_xshift, self.elu_yshift)
            else:
                layer["nonlin"] = self.activation_fn(**self.activation_config)

    def add_first_layer(self):
        layer = OrderedDict()
        layer["conv"] = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.input_kern,
            padding=self.input_kern // 2 if self.pad_input else 0,
            bias=self.bias and not self.batch_norm,
        )
        self.add_bn_layer(layer, self.hidden_channels)
        self.add_activation(layer)
        self.features.add_module("layer0", nn.Sequential(layer))

    def add_subsequent_layers(self):
        if not isinstance(self.hidden_kern, Iterable):
            self.hidden_kern = [self.hidden_kern] * (self.num_layers - 1)

        for l in range(1, self.num_layers):
            layer = OrderedDict()
            if self.hidden_padding is None:
                self.hidden_padding = (
                    (self.hidden_kern[l - 1] - 1) * self.hidden_dilation + 1
                ) // 2
            layer[self.conv_layer_name] = self.ConvLayer(
                in_channels=self.hidden_channels
                if not self.skip > 1
                else min(self.skip, l) * self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.hidden_kern[l - 1],
                stride=self.stride,
                padding=self.hidden_padding,
                dilation=self.hidden_dilation,
                bias=self.bias,
            )
            self.add_bn_layer(layer, self.hidden_channels)
            self.add_activation(layer)
            if l != self.num_layers - 1:
                layer["dropout"] = nn.Dropout2d(p=self.dropout_rate, inplace=True)
            self.features.add_module("layer{}".format(l), nn.Sequential(layer))

    class AttentionConvWrapper(AttentionConv):
        def __init__(self, dilation=None, **kwargs):
            """
            Helper class to make an attention conv layer accept input args of a pytorch.nn.Conv2d layer.
            Args:
                dilation: catches this argument from the input args, and ignores it
                **kwargs:
            """
            super().__init__(**kwargs)

    def initialize(self):
        """Initialization applied on the core."""
        self.apply(self.init_conv)

    @staticmethod
    def init_conv(m):
        """
        Initialize convolution layers with:
            - weights: xavier_normal
            - biases: 0

        Args:
            m (nn.Module): a pytorch nn module.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def laplace(self):
        """
        Laplace regularization for the filters of the first conv2d layer.
        """
        return self._input_weights_regularizer(
            self.features[0].conv.weight, avg=self.use_avg_reg
        )

    def group_sparsity(self):
        """
        Sparsity regularization on the filters of all the Conv2d layers except
        the first one.
        """
        ret = 0
        if self.ignore_group_sparsity:
            return ret

        for feature in self.features[1:]:
            ret = (
                ret
                + feature.conv.weight.pow(2)
                .sum(3, keepdim=True)
                .sum(2, keepdim=True)
                .sqrt()
                .mean()
            )
        return ret / ((self.num_layers - 1) if self.num_layers > 1 else 1)

    def regularizer(self):
        term1 = self.gamma_hidden * self.group_sparsity()
        term2 = self.gamma_input * self.laplace()
        return term1 + term2

    def forward(self, inputs: torch.Tensor):
        outputs = []
        for layer, fn in enumerate(self.features):
            do_skip = layer >= 1 and self.skip > 1
            inputs = fn(
                inputs
                if not do_skip
                else torch.cat(outputs[-min(self.skip, layer) :], dim=1)
            )
            outputs.append(inputs)
        return torch.cat([outputs[ind] for ind in self.stack], dim=1)
