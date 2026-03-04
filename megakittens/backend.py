from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping

import torch
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd


def megakittens_backend(
    fn: Callable[..., Any],
    *,
    verify: bool = False,
    profile: bool = False,
    debug: bool = False,
    save_dag: bool = False,
) -> Callable[[torch.fx.GraphModule, List[Any]], Callable[..., Any]]:
    def _megakittens_backend(gm: torch.fx.GraphModule, example_inputs: List[Any]) -> Callable[..., Any]:
        if debug:
            print(f"[MegaKittens] Compiling function `{fn.__qualname__}`")

            print(f"[MegaKittens] FX graph:")
            gm.graph.print_tabular()
        return make_boxed_func(gm.forward)

    return aot_autograd(
        fw_compiler=_megakittens_backend,
        bw_compiler=_megakittens_backend,
    )
