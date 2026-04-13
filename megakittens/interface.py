from __future__ import annotations

import functools
from typing import Any, Callable

import torch

from .backend import megakittens_backend


def compile(
    fn: Callable[..., Any] | None = None,
    *,
    enable: bool = True,
    verify: bool = False,
    debug: bool = False,
    save_dag: bool = False,
    save_schedule: bool = False,
    use_jit_cache: bool = True,
) -> Callable[..., Any]:
    """Compile a PyTorch function into a MegaKernel."""
    def _compile(_fn: Callable[..., Any]) -> Callable[..., Any]:
        megakernel_fn = torch._dynamo.optimize(
            backend=megakittens_backend(
                fn=_fn,
                verify=verify,
                debug=debug,
                save_dag=save_dag,
                save_schedule=save_schedule,
                use_jit_cache=use_jit_cache,
            ),
            nopython=True, # graph breaks currently not supported (TODO: support it)
            disable=not enable,
            dynamic=False, # dynamic shapes not supported (TODO: support it)
        )(_fn)
        functools.update_wrapper(megakernel_fn, _fn)
        return megakernel_fn

    if fn is not None:
        return _compile(fn)
    else:
        return _compile
