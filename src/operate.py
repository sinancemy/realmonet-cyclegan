import torch


def train(model, dl_A, dl_B):
    pass


def convert(model, dl_A=None, dl_B=None):
    if dl_A is None and dl_B is not None:
        # Run A -> B generator.
        return None
    elif dl_A is not None and dl_B is None:
        # Run B -> A generator.
        return None
    else:
        raise Exception
