"""Profile megakernel decode and export .mkprof for the PIXI.js viewer.
(see megakernels/tp_generate.py in megakernels high throughput)

Usage:
    PYTHONPATH=. python tests/llama1b/profile_decode.py
    PYTHONPATH=. python tests/llama1b/profile_decode.py seq_len=512
"""
import math
import pydra, torch
from megakittens.dispatcher import Dispatcher, ScalarField
from megakittens.jit.cuda_utils import get_sm_count, initialize_cuda_context
from megakittens.llama1b.scheduler import schedule_decode

NL, HD, HDDIM, NKV, NQH = 16, 2048, 64, 8, 32
INTD, VS, MSL = 8192, 128256, 512

SCALARS = [
    ScalarField("pos_id", "unsigned int", 4, 4),
    ScalarField("attn_scale", "float", 4, 4),
    ScalarField("rms_norm_eps", "float", 4, 4),
]


def make_tensors():
    D, bf, fp = "cuda", torch.bfloat16, torch.float32
    r = lambda *s, d=bf: torch.randn(*s, dtype=d, device=D)
    z = lambda *s, d=bf: torch.zeros(*s, dtype=d, device=D)
    QKV = (NQH + 2 * NKV) * HDDIM
    return [
        r(NL, QKV, HD), r(NL, HD, HD), r(NL, HD), r(NL, HD),
        r(NL, INTD, HD), r(NL, INTD, HD), r(NL, HD, INTD),
        r(HD), r(VS, HD), r(HD),
        z(NQH * HDDIM), z(HD), z(INTD), z(VS),
        r(NL, MSL, NKV, HDDIM), r(NL, MSL, NKV, HDDIM),
        r(MSL, HDDIM, d=fp), r(MSL, HDDIM, d=fp),
    ]


class ScriptConfig(pydra.Config):
    seq_len: int = 256
    clock_mhz: float = 2100.0
    num_warmup: int = 3
    name: str = "llama1b_decode"


def main(config: ScriptConfig):
    initialize_cuda_context()
    sm_count = get_sm_count()
    metas, tmetas, insts, nbar, inp, oi = schedule_decode(sm_count=sm_count)
    d = Dispatcher(metas, tmetas, insts, nbar, inp, oi,
                   use_jit_cache=False, scalar_fields=SCALARS)
    tensors = make_tensors()
    pos_id = config.seq_len - 1
    attn_scale = 1.0 / math.sqrt(HDDIM)

    for _ in range(config.num_warmup):
        d(*tensors, pos_id, attn_scale, 1e-5)
    torch.cuda.synchronize()
    d(*tensors, pos_id, attn_scale, 1e-5)
    torch.cuda.synchronize()


if __name__ == "__main__":
    pydra.run(main)
