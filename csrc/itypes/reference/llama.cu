#include "attention_decode.cu"
#include "attention_prefill.cu"
#include "batched_rms_norm.cu"
#include "gate_silu.cu"
#include "inc_barriers.cu"
#include "llama.cuh"
#include "lm_head.cu"
#include "matmul_adds.cu"
#include "pyutils/pyutils.cuh"
#include "qkv_rope_append.cu"
#include "up_matmul.cu"
#include "all_device_barrier.cu"

using namespace kittens;
using namespace megakernel;

struct ops {
    using attn_norm_op = attn_norm<llama_config, llama_70b_globals>;
    using qkv_rope_append_op = qkv_rope_append<llama_config, llama_70b_globals>;
    using attention_prefill_op = attention_prefill<llama_config, llama_70b_globals>;
    using attention_decode_op = attention_decode<llama_config, llama_70b_globals>;
    using o_proj_op = o_proj<llama_config, llama_70b_globals>;

    using mlp_norm_op = mlp_norm<llama_config, llama_70b_globals>;
    using gate_silu_op = gate_silu<llama_config, llama_70b_globals>;
    using up_matmul_op = up_matmul<llama_config, llama_70b_globals>;
    using downproj_op = downproj<llama_config, llama_70b_globals>;

    using lm_head_norm_op = lm_head_norm<llama_config, llama_70b_globals>;
    using lm_head_op = lm_head<llama_config, llama_70b_globals>;

    using barrier_inc_op = barrier_inc<llama_config, llama_70b_globals>;

    using all_device_barrier_op = all_device_barrier<llama_config, llama_70b_globals>;
};

struct ops_timer {
    using attn_norm_op = attn_norm<llama_config_timer, llama_70b_globals_timer>;
    using qkv_rope_append_op = qkv_rope_append<llama_config_timer, llama_70b_globals_timer>;
    using attention_prefill_op = attention_prefill<llama_config_timer, llama_70b_globals_timer>;
    using attention_decode_op = attention_decode<llama_config_timer, llama_70b_globals_timer>;
    using o_proj_op = o_proj<llama_config_timer, llama_70b_globals_timer>;

    using mlp_norm_op = mlp_norm<llama_config_timer, llama_70b_globals_timer>;
    using gate_silu_op = gate_silu<llama_config_timer, llama_70b_globals_timer>;
    using up_matmul_op = up_matmul<llama_config_timer, llama_70b_globals_timer>;
    using downproj_op = downproj<llama_config_timer, llama_70b_globals_timer>;

    using lm_head_norm_op = lm_head_norm<llama_config_timer, llama_70b_globals_timer>;
    using lm_head_op = lm_head<llama_config_timer, llama_70b_globals_timer>;

    using barrier_inc_op = barrier_inc<llama_config_timer, llama_70b_globals_timer>;

    using all_device_barrier_op = all_device_barrier<llama_config_timer, llama_70b_globals_timer>;
};

#define OPS_LIST(ops_type)                                                                                          \
    typename ops_type::attn_norm_op, typename ops_type::qkv_rope_append_op, typename ops_type::attention_decode_op, \
        typename ops_type::attention_prefill_op, typename ops_type::o_proj_op, typename ops_type::mlp_norm_op,      \
        typename ops_type::gate_silu_op, typename ops_type::up_matmul_op, typename ops_type::downproj_op,           \
        typename ops_type::lm_head_norm_op, typename ops_type::lm_head_op, typename ops_type::barrier_inc_op,      \
        typename ops_type::all_device_barrier_op                                                                  \

#define GLOB_ARGS_LIST(globs)                                                                         \
    &globs::Bar, &globs::instructions, &globs::timings, &globs::global_instruction_index,             \
                                                                                                      \
        &globs::qkv_weights, &globs::attn_norm_weights, &globs::o_weights, &globs::mlp_norm_weights,  \
                                                                                                      \
        &globs::up_weights, &globs::gate_weights, &globs::down_weights, &globs::lm_head_norm_weights, \
        &globs::lm_head_weights,                                                                      \
                                                                                                      \
        &globs::k_cache, &globs::v_cache, &globs::rope_cos, &globs::rope_sin,                         \
                                                                                                      \
        &globs::hidden_states, &globs::rms_rope_intermediates, &globs::rms_gate_intermediates,        \
                                                                                                      \
        &globs::q_post_rope, &globs::attn_out, &globs::silu_out,                                      \
                                                                                                      \
        &globs::rms_lm_head_intermediates, &globs::logits,                                            \
                                                                                                      \
        &globs::position_ids, &globs::kv_append_indices,                                              \
                                                                                                      \
        &globs::prefill_qo_indptr, &globs::prefill_kv_indptr, &globs::prefill_kv_indices,             \
        &globs::prefill_kv_last_page_len,                                                             \
                                                                                                      \
        &globs::decode_kv_indptr, &globs::decode_kv_indices, &globs::decode_kv_last_page_len,         \
                                                                                                      \
        &globs::attn_scale, &globs::rms_norm_eps, &globs::num_pages, &globs::batch_size, &globs::num_prefill_tokens

#define COMPILE_TIMINGS

#define COMPILE_MULTIGPU

// #define COMPILE_SINGLE_DEVICE

PYBIND11_MODULE(mk_llama_tp, m) {
    m.doc() = "";
    kittens::py::bind_multigpu_boilerplate(m);

#ifdef COMPILE_MULTIGPU

#ifdef COMPILE_TIMINGS
    kittens::py::bind_multigpu_kernel<mk<llama_config_timer, llama_70b_globals_timer, OPS_LIST(ops_timer)> >(
        m, "mk_llama_tp_timer", GLOB_ARGS_LIST(llama_70b_globals_timer));

#endif

    kittens::py::bind_multigpu_kernel<mk<llama_config, llama_70b_globals, OPS_LIST(ops)> >(
        m, "mk_llama_tp", GLOB_ARGS_LIST(llama_70b_globals));

#endif

#ifdef COMPILE_SINGLE_DEVICE

#ifdef COMPILE_TIMINGS
    kittens::py::bind_kernel<mk<llama_config_timer, llama_70b_globals_timer, OPS_LIST(ops_timer)> >(
        m, "mk_llama_tp_timer_single_device", GLOB_ARGS_LIST(llama_70b_globals_timer),
        &llama_70b_globals_timer::dev_idx);
#endif

    kittens::py::bind_kernel<mk<llama_config, llama_70b_globals, OPS_LIST(ops)> >(
        m, "mk_llama_tp_single_device", GLOB_ARGS_LIST(llama_70b_globals), &llama_70b_globals::dev_idx);
#endif

#ifdef LLAMA_BROADCAST_LM_HEAD_NORM
    m.def("broadcast_lm_head_norm", []() { return true; });
#else
    m.def("broadcast_lm_head_norm", []() { return false; });
#endif
}
