#pragma once

#include "kittens.cuh"

namespace megakittens {

template <typename Config, typename Globals, int I, int SRC0, int SRC1, int SRC2, int DST>
struct ProjResidual {
    static constexpr int PAGE_BYTES = Config::PAGE_SIZE;

    using row_vec = kittens::sv_bf<I>;
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

    __device__ static __forceinline__ kittens::semaphore &inputs_arrived(state_t<Config> &s) { return s.semaphores()[0]; }

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
            const int row_start   = instruction.indices[0];
            const int total_rows  = instruction.indices[1];
            const int out_start   = instruction.indices[2];
            const int out_count   = instruction.indices[3];
            const int out_total   = instruction.indices[4];
            const int lane        = kittens::laneid();
            const int warp_id     = kittens::warpid();

            const int pid = s.lid_to_pid(0);
            row_vec *rows = reinterpret_cast<row_vec*>(s.pages[pid].data);

            const kittens::bf16 *W = reinterpret_cast<const kittens::bf16 *>(
                g.template gls<SRC1>().raw_ptr);
            const kittens::bf16 *residual = reinterpret_cast<const kittens::bf16 *>(
                g.template gls<SRC2>().raw_ptr);
            kittens::bf16 *output = (kittens::bf16 *)g.template gls<DST>().raw_ptr;
            const auto &x_gl = g.template gls<SRC0>();

            const int elems_per_iter = kittens::WARP_THREADS * 4;

            const int opw = (out_count + Config::NUM_CONSUMER_WARPS - 1) / Config::NUM_CONSUMER_WARPS;
            const int o0  = warp_id * opw;
            const int o1  = min(o0 + opw, out_count);

            if (consumer_group::elect_leader()) {
                all_reuse_barrier_wait<Config>(g, instruction);
            }
            consumer_group::sync(1);

            int sem_phase = 1;
            for (int batch_start = 0; batch_start < total_rows; batch_start += ROWS_PER_PAGE) {
                const int batch_rows = min((int)ROWS_PER_PAGE, total_rows - batch_start);

                if (batch_start > 0) {
                    consumer_group::sync(2);

                    if (consumer_group::elect_leader()) {
                        kittens::tma::expect_bytes(inputs_arrived(s), batch_rows * sizeof(row_vec));
                        for (int i = 0; i < batch_rows; i++) {
                            kittens::tma::load_async(rows[i], x_gl, {row_start + batch_start + i, 0}, inputs_arrived(s));
                        }
                    }
                    kittens::wait(inputs_arrived(s), sem_phase);
                    sem_phase ^= 1;
                }

                for (int r = 0; r < batch_rows; r++) {
                    const kittens::bf16 *xd = rows[r].data;
                    const int global_row = row_start + batch_start + r;

                    for (int oi = o0; oi < o1; oi++) {
                        const int col = out_start + oi;
                        const kittens::bf16 *w_row = W + static_cast<int64_t>(col) * I;

                        float dot = 0.0f;
                        for (int cb = 0; cb < I; cb += elems_per_iter) {
                            const int c = cb + lane * 4;
                            if (c + 3 < I) {
                                uint64_t xp = *reinterpret_cast<const uint64_t *>(&xd[c]);
                                float xa, xb, xc, xd2;
                                unpack_x4(xp, xa, xb, xc, xd2);

                                uint64_t wp = *reinterpret_cast<const uint64_t *>(&w_row[c]);
                                float wa, wb, wc, wd;
                                unpack_x4(wp, wa, wb, wc, wd);

                                dot += xa * wa + xb * wb + xc * wc + xd2 * wd;
                            }
                        }
                        dot = warp_reduce_sum(dot);

                        if (lane == 0) {
                            float res = __bfloat162float(
                                residual[static_cast<int64_t>(global_row) * out_total + col]);
                            output[static_cast<int64_t>(global_row) * out_total + col]
                                = __float2bfloat16(dot + res);
                        }
                    }
                }
            }

            __threadfence();
            consumer_group::sync(3);

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
