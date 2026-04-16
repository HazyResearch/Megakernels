#pragma once

#include "kittens.cuh"

namespace megakittens {

enum class BinaryOp { ADD, SUB, MUL, DIV, MAX, MIN, ATAN2 };

template <BinaryOp op, typename T>
__device__ static __forceinline__ void apply_binary(T &dst, const T &a, const T &b) {
    if constexpr (op == BinaryOp::ATAN2) {
        dst = __floats2bfloat162_rn(
            atan2f(__bfloat162float(a.x), __bfloat162float(b.x)), 
            atan2f(__bfloat162float(a.y), __bfloat162float(b.y))
        );
    }
}

template <BinaryOp op, typename Group, kittens::ducks::rt::all RT>
__device__ static __forceinline__ void apply_binary_op(RT &dst, const RT &a, const RT &b) {
    if      constexpr (op == BinaryOp::ADD) Group::add(dst, a, b);
    else if constexpr (op == BinaryOp::SUB) Group::sub(dst, a, b);
    else if constexpr (op == BinaryOp::MUL) Group::mul(dst, a, b);
    else if constexpr (op == BinaryOp::DIV) Group::div(dst, a, b);
    else if constexpr (op == BinaryOp::MAX) Group::max(dst, a, b);
    else if constexpr (op == BinaryOp::MIN) Group::min(dst, a, b);
    else {
        #pragma unroll
        for (int i = 0; i < RT::height; i++)
            #pragma unroll
            for (int j = 0; j < RT::width; j++)
                #pragma unroll
                for (int k = 0; k < RT::packed_per_tile; k++)
                    apply_binary<op>(dst.tiles[i][j].data[k], a.tiles[i][j].data[k], b.tiles[i][j].data[k]);
    }
}

// Get I-th value from an int parameter pack
template <int I, int First, int... Rest>
struct nth_int { static constexpr int value = nth_int<I - 1, Rest...>::value; };
template <int First, int... Rest>
struct nth_int<0, First, Rest...> { static constexpr int value = First; };

// Get last value from an int parameter pack
template <int... Is>
struct last_int;
template <int Last>
struct last_int<Last> { static constexpr int value = Last; };
template <int First, int... Rest>
struct last_int<First, Rest...> { static constexpr int value = last_int<Rest...>::value; };

template <BinaryOp... Ops>
struct BinaryOps {
    static constexpr int NUM_OPS = sizeof...(Ops);
    template <int I>
    __device__ static constexpr BinaryOp get() {
        constexpr BinaryOp arr[] = {Ops...};
        return arr[I];
    }
};

// Example: ElementwiseBinary<Config, Globals, BinaryOps<Op0, Op1, ...>, SRC0, SRC1, ..., SRCN, DST>
template <typename Config, typename Globals, typename Ops, int... TensorIndices>
struct ElementwiseBinary {
    static constexpr int NUM_OPS = Ops::NUM_OPS;
    static constexpr int NUM_INPUTS = NUM_OPS + 1;
    static_assert(NUM_OPS >= 1);
    static_assert(sizeof...(TensorIndices) == NUM_INPUTS + 1, "Need N + 1 tensor indices for N ops (N inputs + 1 output)");
    static constexpr int DST = last_int<TensorIndices...>::value;
    static constexpr int MAX_TILES_PER_INST = 2;
    static constexpr int TILES_PER_INST = (Config::NUM_PAGES / NUM_INPUTS < MAX_TILES_PER_INST) ? Config::NUM_PAGES / NUM_INPUTS : MAX_TILES_PER_INST;
    static_assert(TILES_PER_INST >= 1, "Not enough pages for this many inputs");
    static constexpr int DST_COORD_BASE_IDX = NUM_INPUTS * 4;
    static constexpr int NUM_TILES_IDX = (NUM_INPUTS + 1) * 4;
    static_assert(NUM_TILES_IDX < 16, "Instruction indices slot overflow for this NUM_INPUTS");

    using tile_t = kittens::st<kittens::bf16, 128, 128>;

    __device__ static __forceinline__ kittens::semaphore &inputs_arrived(state_t<Config> &s, int i) { return s.semaphores()[i]; }

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            const int num_tiles = s.instruction().indices[NUM_TILES_IDX];
            const int num_unused = Config::NUM_PAGES - num_tiles*NUM_INPUTS;
            if (query < num_unused)
                return num_tiles * NUM_INPUTS + query;
            const int used_query = query - num_unused;
            constexpr int non_dst_pages = NUM_INPUTS - 1;
            if (used_query < num_tiles * non_dst_pages) {
                const int tile_idx = used_query / non_dst_pages;
                const int input_idx = used_query % non_dst_pages + 1;
                return tile_idx * NUM_INPUTS + input_idx;
            } else {
                const int tile_idx = used_query - num_tiles * non_dst_pages;
                return tile_idx * NUM_INPUTS;
            }
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            const int num_tiles = s.instruction().indices[NUM_TILES_IDX];
            if (kittens::laneid() < num_tiles)
                kittens::init_semaphore(inputs_arrived(s, kittens::laneid()), 1);
            return num_tiles;
        }
    };

    struct loader {
        template <int I>
        __device__ __forceinline__ static void load_tile(const Globals &g, state_t<Config> &s, int tile_idx) {
            const auto &instruction = s.instruction();
            const int batch = instruction.indices[I*4 + 0];
            const int depth = instruction.indices[I*4 + 1];
            const int tile_row = instruction.indices[I*4 + 2];
            const int tile_col_start = instruction.indices[I*4 + 3];
            const int pid = s.lid_to_pid(tile_idx*NUM_INPUTS + I);
            s.page_wait(pid);
            tile_t &tile = s.pages[pid].template as<tile_t>();
            kittens::tma::load_async(tile, g.template gls<nth_int<I, TensorIndices...>::value>(), {batch, depth, tile_row, tile_col_start + tile_idx}, inputs_arrived(s, tile_idx));
        }

        template <int... Is>
        __device__ __forceinline__ static void load_all_tiles(const Globals &g, state_t<Config> &s, int tile_idx, std::integer_sequence<int, Is...>) {
            (load_tile<Is>(g, s, tile_idx), ...);
        }

        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const int num_tiles = instruction.indices[NUM_TILES_IDX];

            if (kittens::warp::elect_leader()) {
                all_input_barrier_wait<Config>(g, instruction);
                for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
                    kittens::tma::expect_bytes(inputs_arrived(s, tile_idx), NUM_INPUTS*sizeof(tile_t)); // TODO: try optimizing with more fine-grained mbarrier arrivals
                    load_all_tiles(g, s, tile_idx, std::make_integer_sequence<int, NUM_INPUTS>{});
                }
            } else if (kittens::warp::elect_leader_from_active()) {
                for (int i = num_tiles*NUM_INPUTS; i < Config::NUM_PAGES; i++) {
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
        using reg_tile_t = kittens::rt_bf<16, 128>;

        template <int... Is>
        __device__ __forceinline__ static void apply_all_binary_ops(reg_tile_t *reg_tiles, std::integer_sequence<int, Is...>) {
            (apply_binary_op<Ops::template get<Is>(), consumer_group>(reg_tiles[0], reg_tiles[0], reg_tiles[Is + 1]), ...);
        }

        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            auto &dst_gl = g.template gls<DST>();
            const int dst_batch = instruction.indices[DST_COORD_BASE_IDX + 0];
            const int dst_depth = instruction.indices[DST_COORD_BASE_IDX + 1];
            const int dst_tile_row = instruction.indices[DST_COORD_BASE_IDX + 2];
            const int dst_tile_col_start = instruction.indices[DST_COORD_BASE_IDX + 3];
            const int num_tiles = instruction.indices[NUM_TILES_IDX];

            for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
                kittens::wait(inputs_arrived(s, tile_idx), 0);

                reg_tile_t reg_tiles[NUM_INPUTS];
                #pragma unroll
                for (int i = 0; i < NUM_INPUTS; i++) {
                    tile_t &tile = s.pages[s.lid_to_pid(tile_idx*NUM_INPUTS + i)].template as<tile_t>();
                    consumer_group::load(reg_tiles[i], tile);
                }

                apply_all_binary_ops(reg_tiles, std::make_integer_sequence<int, NUM_OPS>{});

                tile_t &dst_tile = s.pages[s.lid_to_pid(tile_idx*NUM_INPUTS)].template as<tile_t>();
                consumer_group::store(dst_tile, reg_tiles[0]);
                consumer_group::sync(1);

                if (consumer_group::elect_leader()) {
                    #pragma unroll
                    for (int i = 1; i < NUM_INPUTS; i++)
                        s.page_finish(s.lid_to_pid(tile_idx*NUM_INPUTS + i));
                    if (tile_idx == 0) all_reuse_barrier_wait<Config>(g, instruction);
                    kittens::tma::store_async(dst_gl, dst_tile, {dst_batch, dst_depth, dst_tile_row, dst_tile_col_start + tile_idx});
                }
            }

            if (consumer_group::elect_leader()) {
                kittens::tma::store_async_wait();
                for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) s.page_finish(s.lid_to_pid(tile_idx*NUM_INPUTS));
                all_barrier_arrive<Config>(g, instruction);
            }
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) { }
    };
};

} // namespace megakittens
