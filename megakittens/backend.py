from typing import Any, Callable, List

import torch


def megakittens_backend(
    fn: Callable[..., Any],
    *,
    verify: bool = False,
    profile: bool = False,
    debug: bool = False,
) -> Callable[[torch.fx.GraphModule, List[Any]], Callable[..., Any]]:
    def _megakittens_backend(gm: torch.fx.GraphModule, example_inputs: List[Any]) -> Callable[..., Any]:
        if debug:
            print(f"[MegaKittens] Compiling function `{fn.__qualname__}`")
        return gm.forward

    return _megakittens_backend
