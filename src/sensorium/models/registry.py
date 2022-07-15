import os
import torchinfo

from sensorium.utils import tensorboard

_MODELS = dict()


def register(name):
    """Note: update __init__.py so that register works"""

    def add_to_dict(fn):
        global _MODELS
        _MODELS[name] = fn
        return fn

    return add_to_dict


def get_model(args, summary: tensorboard.Summary = None):
    """Initialize and return model specified in args.model"""
    if not args.model in _MODELS.keys():
        raise NotImplementedError(f"model {args.model} has not been implemented.")

    model = _MODELS[args.model](args)
    model.to(args.device)

    # get model summary and write to args.output_dir
    model_info = torchinfo.summary(
        model, input_size=args.input_shape, device=args.device, verbose=0
    )
    with open(os.path.join(args.output_dir, "model.txt"), "w") as file:
        file.write(str(model_info))
    if args.verbose == 2:
        print(str(model_info))
    if summary is not None:
        summary.scalar("model/trainable_parameters", model_info.trainable_params)

    return model
