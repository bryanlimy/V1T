import torch
import typing as t
from torch import nn
from torch.nn import ModuleDict
from torchvision import transforms


class MLP(nn.Module):
    def __init__(
        self,
        args,
        in_features: int = 2,
        hidden_features: int = 10,
        num_layers: int = 1,
        name: str = "PupilShifter",
    ):
        """
        Multi-layer perceptron shifter
        Args:
            input_features (int): number of input features, defaults to 2.
            hidden_channels (int): number of hidden units.
            shift_layers(int): number of shifter layers (n=1 will correspond
                to a network without a hidden layer).
            **kwargs:
        """
        super(MLP, self).__init__()
        self.name = name
        self.device = args.device
        self.reg_scale = torch.tensor(args.shifter_reg_scale, device=self.device)
        out_features = in_features
        layers = []
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_features=out_features, out_features=hidden_features),
                    nn.Tanh(),
                ]
            )
            out_features = hidden_features
        layers.extend([nn.Linear(in_features=out_features, out_features=2), nn.Tanh()])
        self.mlp = nn.Sequential(*layers)

    def regularizer(self):
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())

    def forward(self, pupil_center: torch.Tensor, trial_idx: torch.Tensor = None):
        if trial_idx is not None:
            pupil_center = torch.cat((pupil_center, trial_idx), dim=1)
        return self.mlp(pupil_center)


from torch.utils.data import DataLoader


class Cropper(nn.Module):
    def __init__(self, args, ds: t.Dict[int, DataLoader]):
        super().__init__()
        mouse_ids = list(ds.keys())
        c, h, w = ds[mouse_ids[0]].dataset.image_shape

        self.crop_scale = args.center_crop
        self.crop_h, self.crop_w = h, w
        if self.crop_scale < 1:
            self.crop_h = int(h * self.crop_scale)
            self.crop_w = int(w * self.crop_scale)
        # the top left coordinate for a center crop
        self.center_coordinate = (h // 2 - self.crop_h // 2, (w - self.crop_w))

        if args.resize_image == 1:
            h, w = 36, 64
        self.resize = transforms.Resize(size=(h, w), antialias=False)
        self.output_shape = (c, h, w)

    def crop_image(
        self,
        images: torch.Tensor,
        coordinate: t.Union[t.Tuple[int, int], torch.Tensor] = None,
    ):
        """Given top left coordinate, crop images to size (self.crop_h, self.crop_w)"""
        if coordinate is None:
            coordinate = self.center_coordinate
        return images[
            ...,
            coordinate[0] : coordinate[0] + self.crop_h,
            coordinate[1] : coordinate[1] + self.crop_w,
        ]

    def forward(self, inputs: torch.Tensor, mouse_id: int, pupil_center: torch.Tensor):
        outputs = self.crop_image(inputs)
        outputs = self.resize(outputs)
        return outputs
