# Placeholder — will be implemented as a decode instruction

import torch


@torch.library.custom_op("megakittens::rms_upgate_silu", mutates_args=())
def rms_upgate_silu_op(
    x: torch.Tensor, norm_weight: torch.Tensor,
    gate_weight: torch.Tensor, up_weight: torch.Tensor, eps: float,
) -> torch.Tensor:
    h = torch.rms_norm(x, [x.shape[-1]], norm_weight, eps)
    return torch.nn.functional.silu(h @ gate_weight.T) * (h @ up_weight.T)


@rms_upgate_silu_op.register_fake
def _fake(x, norm_weight, gate_weight, up_weight, eps):
    return torch.empty((*x.shape[:-1], gate_weight.shape[0]), dtype=x.dtype, device=x.device)
