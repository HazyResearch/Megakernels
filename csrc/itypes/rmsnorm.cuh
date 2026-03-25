
#pragma once

#include "kittens.cuh"

namespace megakittens {

// ── RMSNorm instruction type (TMA sv_bf loads, 1 page) ─────────────────────
//
// Template parameter N is the column width (compile-time).
// One instruction processes multiple rows within a single page.
//
// instruction.indices layout:
//   [0] = starting row index
//   [1] = rows_this_instruction
//
// SRC0 = x      (M, N) bf16
// SRC1 = weight (N,)   bf16
// DST  = output (M, N) bf16

template <typename Config, typename Globals, int N, int SRC0, int SRC1, int DST>
struct RMSNorm {
    static constexpr int PAGE_BYTES = Config::PAGE_SIZE;
    static constexpr float EPS = 1e-6f;

    using row_vec = kittens::sv_bf<N>;
    static constexpr int ROW_BYTES = sizeof(row_vec);
    static constexpr int ROWS_PER_PAGE = PAGE_BYTES / ROW_BYTES;

    __device__ static __forceinline__ float warp_reduce_sum(float val) {
        #pragma unroll
        for (int offset = kittens::WARP_THREADS / 2; offset > 0; offset >>= 1)
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
        return val;
    }

    __device__ static __forceinline__ void unpack_x4(uint64_t packed, float &a, float &b, float &c, float &d) {
        uint32_t lo = static_cast<uint32_t>(packed);
        uint32_t hi = static_cast<uint32_t>(packed >> 32);
        a = __bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>(lo)));
        b = __bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>(lo >> 16)));
        c = __bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>(hi)));
        d = __bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>(hi >> 16)));
    }

    __device__ static __forceinline__ uint64_t pack_x4(float a, float b, float c, float d) {
        uint32_t lo = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(a)))
                    | (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(b))) << 16);
        uint32_t hi = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(c)))
                    | (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(d))) << 16);
        return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
    }

    __device__ static __forceinline__ kittens::semaphore &inputs_arrived(state_t<Config> &s) { return s.semaphores()[0]; }

    // ── controller ─────────────────────────────────────────────────────────
    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int lid) {
            // Only page 0 is used. Release pages 1..NUM_PAGES-1 first, page 0 last.
            if (lid < Config::NUM_PAGES - 1)
                return lid + 1;
            return 0;
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            if (kittens::warp::elect_leader()) {
                kittens::init_semaphore(inputs_arrived(s), 1);
            }
            return 1;
        }
    };

    // ── loader (TMA sv loads) ──────────────────────────────────────────────
    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const int row_start = instruction.indices[0];
            const int num_rows  = instruction.indices[1];

            if (kittens::warp::elect_leader()) {
                all_barrier_wait<Config>(g, instruction);
                const int pid = s.lid_to_pid(0);
                s.page_wait(pid);

                const auto &x_gl = g.template gls<SRC0>();
                row_vec *rows = reinterpret_cast<row_vec*>(s.pages[pid].data);

                kittens::tma::expect_bytes(inputs_arrived(s), num_rows * sizeof(row_vec));
                for (int i = 0; i < num_rows; i++) {
                    kittens::tma::load_async(rows[i], x_gl, {row_start + i, 0}, inputs_arrived(s));
                }
            } else if (kittens::warp::elect_leader_from_active()) {
                // Release unused pages immediately
                for (int i = 1; i < Config::NUM_PAGES; i++) {
                    s.page_wait(s.lid_to_pid(i));
                    s.page_finish(s.lid_to_pid(i));
                }
            }
        }
    };

    // ── launcher ───────────────────────────────────────────────────────────
    struct launcher {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            s.tensor_wait();
            if (kittens::warp::elect_leader()) s.tensor_finish();
        }
    };

    // ── consumer ───────────────────────────────────────────────────────────
    // Row-partitioned: each warp owns ceil(num_rows / NUM_CONSUMER_WARPS) rows.
    // Pass 1: compute sum(x^2) per row -> rstd
    // Pass 2: normalize x[i,j] * rstd[i] * weight[j] -> write back to sv
    // Then TMA store results to global memory.
    struct consumer {
        using consumer_group = kittens::group<Config::NUM_CONSUMER_WARPS>;
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            kittens::wait(inputs_arrived(s), 0);

            const auto &instruction = s.instruction();
            const int row_start = instruction.indices[0];
            const int num_rows  = instruction.indices[1];
            const int lane      = kittens::laneid();
            const int warp_id   = kittens::warpid();

            const int pid = s.lid_to_pid(0);
            row_vec *rows = reinterpret_cast<row_vec*>(s.pages[pid].data);

            // Weight from global memory (shared across all rows, L2-cached)
            const auto &w_gl = g.template gls<SRC1>();
            const kittens::bf16 *weight_ptr = reinterpret_cast<const kittens::bf16 *>(w_gl.raw_ptr);

            // Partition rows across consumer warps
            const int rows_per_warp = (num_rows + Config::NUM_CONSUMER_WARPS - 1) / Config::NUM_CONSUMER_WARPS;
            const int my_row_start  = warp_id * rows_per_warp;
            const int my_row_end    = min(my_row_start + rows_per_warp, num_rows);

            const int elems_per_iter = kittens::WARP_THREADS * 4;

            for (int row = my_row_start; row < my_row_end; row++) {
                kittens::bf16 *row_data = rows[row].data;

                // ── Pass 1: Accumulate sum(x^2) ────────────────────────
                float sum_sq = 0.0f;
                for (int col_base = 0; col_base < N; col_base += elems_per_iter) {
                    const int my_col = col_base + lane * 4;
                    if (my_col + 3 < N) {
                        uint64_t packed = *reinterpret_cast<const uint64_t *>(&row_data[my_col]);
                        float a, b, c, d;
                        unpack_x4(packed, a, b, c, d);
                        sum_sq += a * a + b * b + c * c + d * d;
                    }
                }
                sum_sq = warp_reduce_sum(sum_sq);
                float rstd = rsqrtf(sum_sq / static_cast<float>(N) + EPS);

                // ── Pass 2: Normalize and write back ───────────────────
                for (int col_base = 0; col_base < N; col_base += elems_per_iter) {
                    const int my_col = col_base + lane * 4;
                    if (my_col + 3 < N) {
                        uint64_t packed = *reinterpret_cast<const uint64_t *>(&row_data[my_col]);
                        float a, b, c, d;
                        unpack_x4(packed, a, b, c, d);

                        uint64_t w_packed = *reinterpret_cast<const uint64_t *>(weight_ptr + my_col);
                        float wa, wb, wc, wd;
                        unpack_x4(w_packed, wa, wb, wc, wd);

                        a = a * rstd * wa;
                        b = b * rstd * wb;
                        c = c * rstd * wc;
                        d = d * rstd * wd;

                        *reinterpret_cast<uint64_t *>(&row_data[my_col]) = pack_x4(a, b, c, d);
                    }
                }
            }

            // Sync all consumer warps before TMA store
            consumer_group::sync(1);

            if (consumer_group::elect_leader()) {
                auto &y_gl = g.template gls<DST>();
                for (int i = 0; i < num_rows; i++) {
                    kittens::tma::store_async(y_gl, rows[i], {row_start + i, 0});
                }
                kittens::tma::store_async_wait();
                s.page_finish(pid);
                all_barrier_arrive<Config>(g, instruction);
            }
        }
    };

    // ── storer (empty — consumer handles TMA stores) ───────────────────────
    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) { }
    };
};

} // namespace megakittens