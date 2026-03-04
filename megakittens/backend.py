import torch


def _megakittens_backend(gm: torch.fx.GraphModule, example_inputs: list):
    return gm.forward
