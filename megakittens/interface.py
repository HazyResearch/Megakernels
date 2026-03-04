import functools
from typing import Any, Callable

import torch

from .backend import megakittens_backend


def compile(
    fn: Callable[..., Any] | None = None,
    *,
    enable: bool = True,
    verify: bool = False,
    profile: bool = False,
    debug: bool = False,
) -> Callable[..., Any]:
    """Compile a PyTorch function into a MegaKernel."""
    def _compile(_fn: Callable[..., Any]) -> Callable[..., Any]:
        megakernel_fn = torch._dynamo.optimize(
            backend=megakittens_backend(
                fn=_fn,
                verify=verify,
                profile=profile,
                debug=debug 
            ),
            nopython=True, # graph breaks currently not supported (TODO: support it)
            disable=not enable,
            dynamic=False, # dynamic shapes not supported (TODO: support it)
        )(_fn)
        functools.update_wrapper(megakernel_fn, _fn)
        return megakernel_fn
    if fn is not None:
        return _compile(fn)
    return _compile
