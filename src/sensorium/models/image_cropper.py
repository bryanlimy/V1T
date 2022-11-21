import torch
import typing as t
from torch import nn
from einops import repeat
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader


class ImageShifter(nn.Module):
    def __init__(
        self,
        args,
        max_shift: float,
        hidden_features: int = 10,
        num_layers: int = 1,
        name: str = "ImageShifter",
    ):
        super(ImageShifter, self).__init__()
        assert 0 <= max_shift <= 1
        self.name = name
        self.device = args.device
        self.register_buffer("max_shift", torch.tensor(max_shift))
        self.register_buffer("reg_scale", torch.tensor(args.shifter_reg_scale))
        out_features = 2
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


class ImageCropper(nn.Module):
    def __init__(
        self,
        args,
        ds: t.Dict[int, DataLoader],
        include_behavior: bool,
        use_shifter: bool,
    ):
        super().__init__()

        self.input_shape = args.input_shape
        c, in_h, in_w = args.input_shape
        out_h, out_w = in_h, in_w

        self.include_behavior = include_behavior
        # include the 3 behaviour data as channel of the image
        if self.include_behavior:
            c += 3

        self.crop_scale = args.center_crop
        self.crop_h, self.crop_w = in_h, in_w
        if self.crop_scale < 1:
            out_h = self.crop_h = int(in_h * self.crop_scale)
            out_w = self.crop_w = int(in_w * self.crop_scale)
        self.build_grid()
        if use_shifter:
            max_shift = 1 - self.crop_scale
            self.add_module(
                name="image_shifter",
                module=nn.ModuleDict(
                    {
                        str(mouse_id): ImageShifter(
                            args,
                            max_shift=max_shift,
                            num_layers=3,
                            name=f"Mouse{mouse_id}ImageShifter",
                        )
                        for mouse_id in list(ds.keys())
                    }
                ),
            )
        else:
            self.image_shifter = None

        self.resize = None
        if args.resize_image == 1:
            out_h, out_w = 36, 64
            self.resize = transforms.Resize(size=(out_h, out_w), antialias=False)

        self.output_shape = (c, out_h, out_w)

    def build_grid(self):
        _, in_h, in_w = self.input_shape
        h_pixels = torch.linspace(-self.crop_scale, self.crop_scale, self.crop_h)
        w_pixels = torch.linspace(-self.crop_scale, self.crop_scale, self.crop_w)
        mesh_x, mesh_y = torch.meshgrid(h_pixels, w_pixels, indexing="ij")
        grid = torch.stack((mesh_x, mesh_y), dim=2)
        grid = torch.flip(grid, dims=(2,))  # grid_sample uses (x, y) coordinates
        grid = grid.unsqueeze(0)
        self.register_buffer("grid", grid)

    def regularizer(self, mouse_id: int):
        return (
            0
            if self.image_shifter is None
            else self.image_shifter[str(mouse_id)].regularizer()
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: int,
        pupil_center: torch.Tensor,
        behavior: torch.Tensor,
    ):
        grid = repeat(self.grid, "1 c h w -> b c h w", b=inputs.size(0))
        if self.image_shifter is not None:
            shifts = self.image_shifter[str(mouse_id)](pupil_center)
            grid = grid + shifts[:, None, None, :]
        outputs = F.grid_sample(inputs, grid=grid, align_corners=True)
        if self.resize is not None:
            outputs = self.resize(outputs)
        if self.include_behavior:
            _, _, h, w = outputs.shape
            behavior = repeat(
                behavior,
                "b d -> b d h w",
                h=outputs.size(2),
                w=outputs.size(3),
            )
            outputs = torch.concat((outputs, behavior), dim=1)
        return outputs, grid
