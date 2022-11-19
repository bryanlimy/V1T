import torch
import typing as t
from torch import nn
from einops import repeat
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(
        self,
        args,
        max_shift: float,
        in_features: int = 2,
        hidden_features: int = 10,
        num_layers: int = 1,
        name: str = "PupilShifter",
    ):
        super(MLP, self).__init__()
        assert 0 <= max_shift <= 1
        self.name = name
        self.device = args.device
        self.max_shift = torch.tensor(max_shift, device=self.device)
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

    def forward(self, pupil_center: torch.Tensor):
        shifts = self.mlp(pupil_center)
        shifts = shifts * self.max_shift
        return shifts


class Cropper(nn.Module):
    def __init__(self, args, ds: t.Dict[int, DataLoader]):
        super().__init__()
        mouse_ids = list(ds.keys())
        self.use_shifter = args.use_shifter
        c, in_h, in_w = args.input_shape
        out_h, out_w = in_h, in_w

        self.crop_scale = args.center_crop
        self.crop_h, self.crop_w = in_h, in_w
        if self.crop_scale < 1:
            out_h = self.crop_h = int(in_h * self.crop_scale)
            out_w = self.crop_w = int(in_w * self.crop_scale)

        if args.use_shifter:
            max_shift = (1 - self.crop_scale) / 2
            self.add_module(
                name="shifter",
                module=nn.ModuleDict(
                    {
                        str(mouse_id): MLP(
                            args,
                            max_shift=max_shift,
                            in_features=2,
                            num_layers=3,
                        )
                        for mouse_id in mouse_ids
                    }
                ),
            )
        else:
            self.shifter = None

        if args.resize_image == 1:
            out_h, out_w = 36, 64
        self.resize = transforms.Resize(size=(out_h, out_w), antialias=False)

        self.output_shape = (c, out_h, out_w)

    def build_grid(
        self,
        input_shape: t.Tuple[int, int, int, int],
        shifts: torch.Tensor = None,
    ):
        b, c, in_h, in_w = input_shape
        h_pixels = torch.linspace(-self.crop_scale, self.crop_scale, self.crop_h)
        w_pixels = torch.linspace(-self.crop_scale, self.crop_scale, self.crop_w)
        mesh_x, mesh_y = torch.meshgrid(h_pixels, w_pixels, indexing="ij")
        grid = torch.stack((mesh_x, mesh_y), dim=2)
        grid = torch.flip(grid, dims=(2,))  # grid_sample uses (x, y) coordinates
        grid = repeat(grid.unsqueeze(0), "1 c h w -> b c h w", b=b)
        if shifts is not None:
            grid = grid + repeat(shifts, "b d -> b 1 1 d")
        return grid

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: int,
        pupil_center: torch.Tensor,
    ):
        shifts = (
            None if self.shifter is None else self.shifter[str(mouse_id)](pupil_center)
        )
        grid = self.build_grid(input_shape=inputs.shape, shifts=shifts)
        outputs = F.grid_sample(inputs, grid=grid, align_corners=True)
        outputs = self.resize(outputs)
        return outputs
