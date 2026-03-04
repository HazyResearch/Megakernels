import functools

import torch

from .backend import _megakittens_backend


def compile(
    fn=None, 
    *,
    enable=True,
    verify=False,
    profile=False,
    debug=False,
):
    """Compile a PyTorch function into a megakernel."""
    def _compile(_fn):
        if debug:
            print(f"[MegaKittens] Compiling function `{_fn.__qualname__}`")
        megakernel = torch._dynamo.optimize(
            backend=_megakittens_backend,
            nopython=True, # graph breaks currently not supported (TODO: support it)
            disable=not enable,
            dynamic=False, # dynamic shapes not supported (TODO: support it)
        )(_fn)
        functools.update_wrapper(megakernel, _fn)
        return megakernel
    if fn is not None:
        return _compile(fn)
    return _compile
