from __future__ import annotations

from typing import Any, Callable, List

import torch
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

from .dag_optimizer import optimize_dag
from .dispatcher import Dispatcher
from .fx_parser import extract_dag_from_fx_graph
from .scheduler import schedule
from .utils import create_log_base_path, save_dag_as_png, save_dag_as_png_as_json, save_schedule_as_txt, timed


def megakittens_backend(
    fn: Callable[..., Any],
    *,
    dry_run: bool = False,
    verify: bool = False,
    profile: bool = False,
    save_dag: bool = False,
    save_schedule: bool = False,
    use_jit_cache: bool = True,
    verbose: bool = True,
    global_work_queue: bool = False,
    cluster_size: int = 2,
) -> Callable[[torch.fx.GraphModule, List[Any]], Callable[..., Any]]:
    def _megakittens_backend(gm: torch.fx.GraphModule, example_inputs: List[Any]) -> Callable[..., Any]:
        if verbose:
            name = getattr(fn, '__qualname__', None) or type(fn).__qualname__
            print(f"[MegaKittens] Compiling `{name}`")
            print(f"[MegaKittens] FX graph:")
            gm.graph.print_tabular()

        if save_dag or save_schedule:
            base_path = create_log_base_path(fn=fn)

        with timed("Built DAG from FX graph", verbose):
            dag = extract_dag_from_fx_graph(gm, example_inputs)

        with timed("Optimized DAG", verbose):
            dag = optimize_dag(dag)

        if save_dag:
            with timed("Saved DAG as JSON", verbose):
                dag_json = save_dag_as_png_as_json(dag, base_path)
            with timed("Saved DAG as PNG", verbose):
                save_dag_as_png(dag_json, base_path)

        if dry_run:
            if verbose:
                print(f"[MegaKittens] Dry run mode; returning original function")
            return make_boxed_func(gm)

        with timed("Scheduled instructions", verbose):
            (
                instruction_metas,
                tensor_metas,
                instructions,
                num_barriers,
                input_tensor_indices,
                output_tensor_indices,
            ) = schedule(dag, cluster_size=cluster_size, verbose=verbose)

        if save_schedule:
            with timed("Saved schedule as TXT", verbose):
                save_schedule_as_txt(
                    tensor_metas, instructions, instruction_metas, num_barriers, base_path
                )

        with timed("Created dispatcher", verbose):
            dispatcher = Dispatcher(
                instruction_metas=instruction_metas,
                tensor_metas=tensor_metas,
                instructions=instructions,
                num_barriers=num_barriers,
                input_tensor_indices=input_tensor_indices,
                output_tensor_indices=output_tensor_indices,
                use_jit_cache=use_jit_cache,
                verbose=verbose,
                global_work_queue=global_work_queue,
                cluster_size=cluster_size,
            )

        return make_boxed_func(dispatcher)

    return aot_autograd(
        fw_compiler=_megakittens_backend,
        bw_compiler=_megakittens_backend,
    )
