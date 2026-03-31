#pragma once

#include "kittens.cuh"

namespace megakittens {

template <typename Config, typename Globals, int N, int SRC0, int SRC1, int SRC2, int SRC3, int DST>
struct RmsUpgateSilu {
    static constexpr int PAGE_BYTES = Config::PAGE_SIZE;
    static constexpr float EPS = 1e-5f;

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

    __device__ static __forceinline__ void normalize_batch(
        row_vec *rows,
        int batch_rows,
        const kittens::bf16 *norm_weight,
        int warp_id,
        int lane
    ) {
        const int elems_per_iter = kittens::WARP_THREADS * 4;
        const int rpw = (batch_rows + Config::NUM_CONSUMER_WARPS - 1) / Config::NUM_CONSUMER_WARPS;
        const int r0  = warp_id * rpw;
        const int r1  = min(r0 + rpw, batch_rows);

        for (int r = r0; r < r1; r++) {
            kittens::bf16 *rd = rows[r].data;

            float sum_sq = 0.0f;
            for (int cb = 0; cb < N; cb += elems_per_iter) {
                const int col = cb + lane * 4;
                if (col + 3 < N) {
                    uint64_t p = *reinterpret_cast<const uint64_t *>(&rd[col]);
                    float a, b, c, d;
                    unpack_x4(p, a, b, c, d);
                    sum_sq += a * a + b * b + c * c + d * d;
                }
            }
            sum_sq = warp_reduce_sum(sum_sq);
            float rstd = rsqrtf(sum_sq / static_cast<float>(N) + EPS);

            for (int cb = 0; cb < N; cb += elems_per_iter) {
                const int col = cb + lane * 4;
                if (col + 3 < N) {
                    uint64_t p = *reinterpret_cast<const uint64_t *>(&rd[col]);
                    float a, b, c, d;
                    unpack_x4(p, a, b, c, d);

                    uint64_t wp = *reinterpret_cast<const uint64_t *>(&norm_weight[col]);
                    float wa, wb, wc, wd;
                    unpack_x4(wp, wa, wb, wc, wd);

                    *reinterpret_cast<uint64_t *>(&rd[col]) = pack_x4(
                        a * rstd * wa, b * rstd * wb,
                        c * rstd * wc, d * rstd * wd);
                }
            }
        }
    }

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int lid) {
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

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const int row_start  = instruction.indices[0];
            const int total_rows = instruction.indices[1];
            const int first_batch = min(total_rows, (int)ROWS_PER_PAGE);

            if (kittens::warp::elect_leader()) {
                all_input_barrier_wait<Config>(g, instruction);
                const int pid = s.lid_to_pid(0);
                s.page_wait(pid);

                const auto &x_gl = g.template gls<SRC0>();
                row_vec *rows = reinterpret_cast<row_vec*>(s.pages[pid].data);

                kittens::tma::expect_bytes(inputs_arrived(s), first_batch * sizeof(row_vec));
                for (int i = 0; i < first_batch; i++) {
                    kittens::tma::load_async(rows[i], x_gl, {row_start + i, 0}, inputs_arrived(s));
                }
            } else if (kittens::warp::elect_leader_from_active()) {
                for (int i = 1; i < Config::NUM_PAGES; i++) {
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
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            kittens::wait(inputs_arrived(s), 0);

            const auto &instruction = s.instruction();
            const int row_start    = instruction.indices[0];
            const int total_rows   = instruction.indices[1];
            const int inter_start  = instruction.indices[2];
            const int inter_count  = instruction.indices[3];
            const int inter_total  = instruction.indices[4];
            const int lane         = kittens::laneid();
            const int warp_id      = kittens::warpid();

            const int pid = s.lid_to_pid(0);
            row_vec *rows = reinterpret_cast<row_vec*>(s.pages[pid].data);

            const kittens::bf16 *norm_weight = reinterpret_cast<const kittens::bf16 *>(
                g.template gls<SRC1>().raw_ptr);
            const kittens::bf16 *up_W = reinterpret_cast<const kittens::bf16 *>(
                g.template gls<SRC2>().raw_ptr);
            const kittens::bf16 *gate_W = reinterpret_cast<const kittens::bf16 *>(
                g.template gls<SRC3>().raw_ptr);
            kittens::bf16 *output = (kittens::bf16 *)g.template gls<DST>().raw_ptr;
            const auto &x_gl = g.template gls<SRC0>();

            const int elems_per_iter = kittens::WARP_THREADS * 4;

            const int ipw = (inter_count + Config::NUM_CONSUMER_WARPS - 1) / Config::NUM_CONSUMER_WARPS;
            const int i0  = warp_id * ipw;
            const int i1  = min(i0 + ipw, inter_count);

            int first_batch = min(total_rows, (int)ROWS_PER_PAGE);
            normalize_batch(rows, first_batch, norm_weight, warp_id, lane);
            consumer_group::sync(1);

            if (consumer_group::elect_leader()) {
                all_reuse_barrier_wait<Config>(g, instruction);
            }
            consumer_group::sync(2);

            int sem_phase = 1;
            for (int batch_start = 0; batch_start < total_rows; batch_start += ROWS_PER_PAGE) {
                const int batch_rows = min((int)ROWS_PER_PAGE, total_rows - batch_start);

                if (batch_start > 0) {
                    consumer_group::sync(3);

                    if (consumer_group::elect_leader()) {
                        kittens::tma::expect_bytes(inputs_arrived(s), batch_rows * sizeof(row_vec));
                        for (int i = 0; i < batch_rows; i++) {
                            kittens::tma::load_async(rows[i], x_gl, {row_start + batch_start + i, 0}, inputs_arrived(s));
                        }
                    }
                    kittens::wait(inputs_arrived(s), sem_phase);
                    sem_phase ^= 1;

                    normalize_batch(rows, batch_rows, norm_weight, warp_id, lane);
                    consumer_group::sync(1);
                }

                for (int r = 0; r < batch_rows; r++) {
                    const kittens::bf16 *xd = rows[r].data;

                    for (int ii = i0; ii < i1; ii++) {
                        const int idx = inter_start + ii;
                        const kittens::bf16 *up_row   = up_W   + static_cast<int64_t>(idx) * N;
                        const kittens::bf16 *gate_row = gate_W + static_cast<int64_t>(idx) * N;

                        float dot_up = 0.0f;
                        float dot_gate = 0.0f;
                        for (int cb = 0; cb < N; cb += elems_per_iter) {
                            const int col = cb + lane * 4;
                            if (col + 3 < N) {
                                uint64_t xp = *reinterpret_cast<const uint64_t *>(&xd[col]);
                                float xa, xb, xc, xd2;
                                unpack_x4(xp, xa, xb, xc, xd2);

                                uint64_t up = *reinterpret_cast<const uint64_t *>(&up_row[col]);
                                float ua, ub, uc, ud;
                                unpack_x4(up, ua, ub, uc, ud);

                                uint64_t gp = *reinterpret_cast<const uint64_t *>(&gate_row[col]);
                                float ga, gb, gc, gd;
                                unpack_x4(gp, ga, gb, gc, gd);

                                dot_up   += xa * ua + xb * ub + xc * uc + xd2 * ud;
                                dot_gate += xa * ga + xb * gb + xc * gc + xd2 * gd;
                            }
                        }
                        dot_up   = warp_reduce_sum(dot_up);
                        dot_gate = warp_reduce_sum(dot_gate);

                        if (lane == 0) {
                            float silu_gate = dot_gate / (1.0f + expf(-dot_gate));
                            output[static_cast<int64_t>(row_start + batch_start + r) * inter_total + idx]
                                = __float2bfloat16(dot_up * silu_gate);
                        }
                    }
                }
            }

            __threadfence();
            consumer_group::sync(4);

            if (consumer_group::elect_leader()) {
                s.page_finish(pid);
                all_barrier_arrive<Config>(g, instruction);
            }
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) { }
    };
};

} // namespace megakittens
