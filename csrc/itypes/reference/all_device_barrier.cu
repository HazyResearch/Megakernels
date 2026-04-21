
#include "llama.cuh"

using namespace kittens;
using namespace megakernel;

template <typename config, typename globals> struct all_device_barrier {
    static constexpr int opcode = OPCODE_AllDeviceBarrier;

    struct parsed_instruction {
        int layer_idx;
        int bar_idx;
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            layer_idx = instruction[1];
            bar_idx = instruction[2];
        }
        __device__ inline parsed_instruction(state<config> &s) : parsed_instruction(s.instruction()) {}
    };

    struct controller {
        static __device__ int
        release_lid(const globals &g,
                    typename config::instruction_t &instruction, int &query) {
            return query;
        }
        static __device__ int init_semaphores(const globals &g,
                                              state<config> &s) {
            return 0;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {

            if (kittens::laneid() < config::NUM_PAGES) { // Release all pages, ASAP.
                auto pid = s.pid(kittens::laneid());
                s.wait_page_ready(pid);
            }

            warp::sync();

            if (kittens::laneid() == 0) {

                parsed_instruction inst(s);

                // sync on device 0
                auto& bar = g.Bar[0][{inst.layer_idx, opcode - 1, inst.bar_idx, 0}];

                redAdd<Sem::RELAXED, Scope::SYS>(&bar, 1);

                auto total_count = g.sm_count * g.num_devices;

                wait_on_barrier<Scope::SYS>(&bar, total_count, "all_device_barrier",
                    "dev=%d (sm_count=%d * num_devices=%d)",
                    g.dev_idx, g.sm_count, g.num_devices);
            }


            warp::sync();

            if (kittens::laneid() < config::NUM_PAGES) { // Release all pages, ASAP.
                auto pid = s.pid(kittens::laneid());
                s.finish_page(pid, config::NUM_CONSUMER_WARPS);
            }

            kittens::warp::arrive(s.instruction_fetch_ready, config::NUM_CONSUMER_WARPS);
        }
    };
    struct launcher { // launches mma's
        // launcher does nothing here, since this doesn't use tensor cores.
        static __device__ void run(const globals &g, state<config> &s) {
#ifdef KITTENS_BLACKWELL
            s.wait_tensor_ready();
            if (kittens::laneid() == 0)
                arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
#endif
        }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {}
    };
    struct storer {
        // Uses 4 full pages for outputs.
        static __device__ void run(const globals &g, state<config> &s) {}
    };
};

