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
        name: str = "MLPShifter",
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
        if self.crop_scale < 1:
            h, w = int(h * self.crop_scale), int(w * self.crop_scale)
        self.center_crop = transforms.CenterCrop(size=(h, w))
        if args.resize_image == 1:
            h, w = 36, 64
        self.resize = transforms.Resize(size=(h, w), antialias=False)
        self.output_shape = (c, h, w)

    def forward(self, inputs: torch.Tensor, mouse_id: int, pupil_center: torch.Tensor):
        outputs = self.center_crop(inputs)
        outputs = self.resize(outputs)
        return outputs
