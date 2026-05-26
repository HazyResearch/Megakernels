#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include "rms.cuh"
#include "qkv_rope_append.cuh"
#include "gate_silu.cuh"
#include "up_matmul.cuh"
#include "o_proj_residual.cuh"
#include "lm_head.cuh"
#include "attention_decode.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_forward",              &manual_kernels::rms_dispatch);
    m.def("qkv_rope_append_forward",  &manual_kernels::qkv_rope_append_dispatch);
    m.def("gate_silu_forward",        &manual_kernels::gate_silu_dispatch);
    m.def("up_matmul_forward",        &manual_kernels::up_matmul_dispatch);
    m.def("o_proj_residual_forward",  &manual_kernels::o_proj_residual_dispatch);
    m.def("lm_head_forward",          &manual_kernels::lm_head_dispatch);
    m.def("attention_decode_forward", &manual_kernels::attention_decode_dispatch);
}
