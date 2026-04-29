#pragma once

#include "kittens.cuh"

namespace megakittens {

enum class UnaryOp { IDENTITY, RELU, ABS, EXP, EXP2, LOG, LOG2, NEG, SQRT, RSQRT, ADD_SCALAR, MUL_SCALAR, SUB_SCALAR, DIV_SCALAR, MAX_SCALAR, MIN_SCALAR };

template <UnaryOp op, unsigned int ScalarBits, typename ElemType, typename Group, kittens::ducks::rt::all RT>
__device__ static __forceinline__ void apply_unary_op(RT &reg) {
    if      constexpr (op == UnaryOp::IDENTITY) {}
    else if constexpr (op == UnaryOp::RELU)  Group::relu(reg, reg);
    else if constexpr (op == UnaryOp::ABS)   Group::abs(reg, reg);
    else if constexpr (op == UnaryOp::EXP)   Group::exp(reg, reg);
    else if constexpr (op == UnaryOp::EXP2)  Group::exp2(reg, reg);
    else if constexpr (op == UnaryOp::LOG)   Group::log(reg, reg);
    else if constexpr (op == UnaryOp::LOG2)  Group::log2(reg, reg);
    else if constexpr (op == UnaryOp::NEG)   Group::neg(reg, reg);
    else if constexpr (op == UnaryOp::SQRT)  Group::sqrt(reg, reg);
    else if constexpr (op == UnaryOp::RSQRT) Group::rsqrt(reg, reg);
    else if constexpr (op == UnaryOp::ADD_SCALAR) Group::add(reg, reg, std::bit_cast<ElemType>(sizeof(ElemType) == 2 ? static_cast<uint16_t>(ScalarBits) : ScalarBits););
    else if constexpr (op == UnaryOp::MUL_SCALAR) Group::mul(reg, reg, std::bit_cast<ElemType>(sizeof(ElemType) == 2 ? static_cast<uint16_t>(ScalarBits) : ScalarBits););
    else if constexpr (op == UnaryOp::SUB_SCALAR) Group::sub(reg, reg, std::bit_cast<ElemType>(sizeof(ElemType) == 2 ? static_cast<uint16_t>(ScalarBits) : ScalarBits););
    else if constexpr (op == UnaryOp::DIV_SCALAR) Group::div(reg, reg, std::bit_cast<ElemType>(sizeof(ElemType) == 2 ? static_cast<uint16_t>(ScalarBits) : ScalarBits););
    else if constexpr (op == UnaryOp::MAX_SCALAR) Group::max(reg, reg, std::bit_cast<ElemType>(sizeof(ElemType) == 2 ? static_cast<uint16_t>(ScalarBits) : ScalarBits););
    else if constexpr (op == UnaryOp::MIN_SCALAR) Group::min(reg, reg, std::bit_cast<ElemType>(sizeof(ElemType) == 2 ? static_cast<uint16_t>(ScalarBits) : ScalarBits););
    else static_assert(false, "Unsupported UnaryOp");
}

template <typename Config, typename Globals, typename ElemType, int SRC, int DST, unsigned int ScalarBits, UnaryOp... Ops>
struct ElementwiseUnary {
    static constexpr int TILE_ROWS = 128;
    static constexpr int TILE_COLS = Config::PAGE_SIZE / (TILE_ROWS * sizeof(ElemType));
    static constexpr int MAX_TILES_PER_INST = 2;
    static constexpr int NUM_USED_PAGES = (Config::NUM_PAGES < MAX_TILES_PER_INST) ? Config::NUM_PAGES : MAX_TILES_PER_INST;

    using tile_t = kittens::st<ElemType, TILE_ROWS, TILE_COLS>;

    __device__ static __forceinline__ kittens::semaphore &inputs_arrived(state_t<Config> &s, int i) { return s.semaphores()[i]; }

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            const int num_tiles = s.instruction().indices[8];
            const int num_unused = Config::NUM_PAGES - num_tiles;
            if (query < num_unused)
                return num_tiles + query;
            else
                return query - num_unused;
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            const int num_tiles = s.instruction().indices[8];
            if (kittens::laneid() < num_tiles)
                kittens::init_semaphore(inputs_arrived(s, kittens::laneid()), 1);
            return num_tiles;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const auto &src_gl = g.template gls<SRC>();
            const int src_batch = instruction.indices[0];
            const int src_depth = instruction.indices[1];
            const int src_tile_row = instruction.indices[2];
            const int src_tile_col_start = instruction.indices[3];
            const int num_tiles = instruction.indices[8];

            if (kittens::warp::elect_leader()) {
                all_input_barrier_wait<Config>(g, instruction);

                for (int i = 0; i < num_tiles; i++) {
                    const int pid = s.lid_to_pid(i);
                    s.page_wait(pid);
                    tile_t &src_smem = s.pages[pid].template as<tile_t>();
                    kittens::tma::expect_bytes(inputs_arrived(s, i), sizeof(tile_t));
                    kittens::tma::load_async(src_smem, src_gl, {src_batch, src_depth, src_tile_row, src_tile_col_start + i}, inputs_arrived(s, i));
                }
            } else if (kittens::warp::elect_leader_from_active()) {
                for (int i = num_tiles; i < Config::NUM_PAGES; i++) {
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
        using reg_tile_t = kittens::rt<ElemType, TILE_ROWS / Config::NUM_CONSUMER_WARPS, TILE_COLS>;

        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            auto &dst_gl = g.template gls<DST>();
            const int dst_batch = instruction.indices[4];
            const int dst_depth = instruction.indices[5];
            const int dst_tile_row = instruction.indices[6];
            const int dst_tile_col_start = instruction.indices[7];
            const int num_tiles = instruction.indices[8];

            for (int t = 0; t < num_tiles; t++) {
                tile_t &src_smem = s.pages[s.lid_to_pid(t)].template as<tile_t>();
                kittens::wait(inputs_arrived(s, t), 0);

                reg_tile_t src_reg;
                consumer_group::load(src_reg, src_smem);
                (apply_unary_op<Ops, ScalarBits, ElemType, consumer_group>(src_reg), ...);
                consumer_group::store(src_smem, src_reg);
                consumer_group::sync(1);

                if (consumer_group::elect_leader()) {
                    if (t == 0) all_reuse_barrier_wait<Config>(g, instruction);
                    kittens::tma::store_async(dst_gl, src_smem, {dst_batch, dst_depth, dst_tile_row, dst_tile_col_start + t});
                }
            }

            if (consumer_group::elect_leader()) {
                kittens::tma::store_async_wait();
                for (int t = 0; t < num_tiles; t++) s.page_finish(s.lid_to_pid(t));
                all_barrier_arrive<Config>(g, instruction);
            }
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) { }
    };
};

} // namespace megakittens
