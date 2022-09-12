_READOUTS = dict()


def register(name):
    """Note: update __init__.py so that register works"""

    def add_to_dict(fn):
        global _READOUTS
        _READOUTS[name] = fn
        return fn

    return add_to_dict


def get_readout(args):
    if not args.readout in _READOUTS.keys():
        raise NotImplementedError(f"Readout {args.readout} has not been implemented.")
    return _READOUTS[args.readout]
