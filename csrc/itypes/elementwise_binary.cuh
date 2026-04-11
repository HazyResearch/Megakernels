#pragma once

#include "kittens.cuh"

namespace megakittens {

enum class BinaryOp { ADD, SUB, MUL, DIV, MAX, MIN };

template <BinaryOp op, typename Group, typename T>
__device__ static __forceinline__ void apply_binary_op(T &dst, const T &a, const T &b) {
    if      constexpr (op == BinaryOp::ADD) Group::add(dst, a, b);
    else if constexpr (op == BinaryOp::SUB) Group::sub(dst, a, b);
    else if constexpr (op == BinaryOp::MUL) Group::mul(dst, a, b);
    else if constexpr (op == BinaryOp::DIV) Group::div(dst, a, b);
    else if constexpr (op == BinaryOp::MAX) Group::max(dst, a, b);
    else if constexpr (op == BinaryOp::MIN) Group::min(dst, a, b);
}

// Get I-th value from an int parameter pack
template <int I, int First, int... Rest>
struct nth_int { static constexpr int value = nth_int<I-1, Rest...>::value; };
template <int First, int... Rest>
struct nth_int<0, First, Rest...> { static constexpr int value = First; };

// Get last value from an int parameter pack
template <int... Is>
struct last_int;
template <int Only>
struct last_int<Only> { static constexpr int value = Only; };
template <int First, int... Rest>
struct last_int<First, Rest...> { static constexpr int value = last_int<Rest...>::value; };

// Type-level list of BinaryOps
template <BinaryOp... Ops>
struct BinaryOpList {
    static constexpr int size = sizeof...(Ops);
    template <int I>
    __host__ __device__ static constexpr BinaryOp get() {
        constexpr BinaryOp arr[] = {Ops...};
        return arr[I];
    }
};

// ElementwiseBinary<Config, Globals, BinaryOpList<Op0, Op1, ...>, SRC0, SRC1, ..., SRCN, DST>
// result = OpN-1(... Op1(Op0(SRC0, SRC1), SRC2) ..., SRCN)
// Tensor indices: first N are sources, last one is DST (matches {tensors} convention)
template <typename Config, typename Globals, typename OpList, int... TensorIndices>
struct ElementwiseBinary {
    static constexpr int N_OPS = OpList::size;
    static constexpr int N_TENSORS = sizeof...(TensorIndices);
    static constexpr int N_INPUTS = N_OPS + 1;
    static_assert(N_TENSORS == N_INPUTS + 1, "Need N+1 tensor indices for N ops (N inputs + 1 output)");

    static constexpr int DST = last_int<TensorIndices...>::value;
    static constexpr int PAGES_PER_TILE = N_INPUTS;
    static constexpr int TILES_PER_INST = Config::NUM_PAGES / PAGES_PER_TILE;
    static_assert(TILES_PER_INST >= 1, "Not enough pages for this many inputs");
    static constexpr int NUM_USED_PAGES = TILES_PER_INST * PAGES_PER_TILE;

    using tile_t = kittens::st<kittens::bf16, 128, 128>;

    __device__ static __forceinline__ kittens::semaphore &inputs_arrived(state_t<Config> &s, int i) { return s.semaphores()[i]; }

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            const int num_tiles = s.instruction().indices[2];
            const int num_unused = Config::NUM_PAGES - num_tiles * PAGES_PER_TILE;
            if (query < num_unused)
                return num_tiles * PAGES_PER_TILE + query;
            const int used_query = query - num_unused;
            // Release in reverse input order: last input first, first input (used for store) last
            const int input_group = (PAGES_PER_TILE - 1) - used_query / num_tiles;
            const int tile_idx = used_query % num_tiles;
            return tile_idx * PAGES_PER_TILE + input_group;
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            if (kittens::laneid() < TILES_PER_INST)
                kittens::init_semaphore(inputs_arrived(s, kittens::laneid()), 1);
            return TILES_PER_INST;
        }
    };

    struct loader {
        template <int I>
        __device__ __forceinline__ static void load_one(const Globals &g, state_t<Config> &s,
                                                         int tile_idx, int tile_row, int tile_col) {
            const int pid = s.lid_to_pid(tile_idx * PAGES_PER_TILE + I);
            s.page_wait(pid);
            tile_t &st = s.pages[pid].template as<tile_t>();
            kittens::tma::load_async(st, g.template gls<nth_int<I, TensorIndices...>::value>(),
                                     {tile_row, tile_col}, inputs_arrived(s, tile_idx));
        }

        template <int... Is>
        __device__ __forceinline__ static void load_all(const Globals &g, state_t<Config> &s,
                                                         int tile_idx, int tile_row, int tile_col,
                                                         std::integer_sequence<int, Is...>) {
            (load_one<Is>(g, s, tile_idx, tile_row, tile_col), ...);
        }

        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const int tile_row = instruction.indices[0];
            const int tile_col_start = instruction.indices[1];
            const int num_tiles = instruction.indices[2];

            if (kittens::warp::elect_leader()) {
                all_input_barrier_wait<Config>(g, instruction);

                for (int i = 0; i < num_tiles; i++) {
                    kittens::tma::expect_bytes(inputs_arrived(s, i), N_INPUTS * sizeof(tile_t));
                    load_all(g, s, i, tile_row, tile_col_start + i, std::make_integer_sequence<int, N_INPUTS>{});
                }
            } else if (kittens::warp::elect_leader_from_active()) {
                for (int i = num_tiles * PAGES_PER_TILE; i < Config::NUM_PAGES; i++) {
                    s.page_wait(s.lid_to_pid(i));
                    s.page_finish(s.lid_to_pid(i));
                }
            }
        }
    };

    struct launcher {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            s.tensor_wait();
            if (kittens::warp::elect_leader()) s.tensor_finish();
        }
    };

    struct consumer {
        using consumer_group = kittens::group<Config::NUM_CONSUMER_WARPS>;
        using reg_t = kittens::rt_bf<16, 128>;

        template <int OpIdx>
        __device__ __forceinline__ static void apply_one(reg_t &acc, reg_t *regs) {
            apply_binary_op<OpList::template get<OpIdx>(), consumer_group>(acc, acc, regs[OpIdx + 1]);
        }

        template <int... OpIs>
        __device__ __forceinline__ static void apply_remaining(reg_t &acc, reg_t *regs, std::integer_sequence<int, OpIs...>) {
            (apply_one<OpIs + 1>(acc, regs), ...); // +1 because ops[0] already applied
        }

        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            auto &dst_gl = g.template gls<DST>();
            const int tile_row = instruction.indices[0];
            const int tile_col_start = instruction.indices[1];
            const int num_tiles = instruction.indices[2];

            for (int t = 0; t < num_tiles; t++) {
                kittens::wait(inputs_arrived(s, t), 0);

                reg_t regs[N_INPUTS];
                #pragma unroll
                for (int i = 0; i < N_INPUTS; i++) {
                    tile_t &st = s.pages[s.lid_to_pid(t * PAGES_PER_TILE + i)].template as<tile_t>();
                    consumer_group::load(regs[i], st);
                }

                // acc = Op0(src0, src1), then acc = OpI(acc, src[I+1]) for remaining
                apply_binary_op<OpList::template get<0>(), consumer_group>(regs[0], regs[0], regs[1]);
                if constexpr (N_OPS > 1) {
                    apply_remaining(regs[0], regs, std::make_integer_sequence<int, N_OPS - 1>{});
                }

                tile_t &dst_st = s.pages[s.lid_to_pid(t * PAGES_PER_TILE)].template as<tile_t>();
                consumer_group::store(dst_st, regs[0]);
                consumer_group::sync(1);

                if (consumer_group::elect_leader()) {
                    #pragma unroll
                    for (int i = N_INPUTS - 1; i >= 1; i--)
                        s.page_finish(s.lid_to_pid(t * PAGES_PER_TILE + i));
                    if (t == 0) all_reuse_barrier_wait<Config>(g, instruction);
                    kittens::tma::store_async(dst_gl, dst_st, {tile_row, tile_col_start + t});
                }
            }

            if (consumer_group::elect_leader()) {
                kittens::tma::store_async_wait();
                for (int t = 0; t < num_tiles; t++) s.page_finish(s.lid_to_pid(t * PAGES_PER_TILE));
                all_barrier_arrive<Config>(g, instruction);
            }
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) { }
    };
};

} // namespace megakittens
