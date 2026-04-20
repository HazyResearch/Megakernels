#pragma once

// Minimal device-side PGL type that matches the Python `pgl.memory_layout`
// byte-for-byte for the subset of kittens::pgl we actually use from device
// code (per-peer `gls[i]` access only — no multicast).
//
// We don't depend on kittens::pgl from the TK megakernels branch because
// that branch's <stdint.h>/<string>/<type_traits> includes trip NVRTC
// header-resolution under CUDA 13 CCCL. Matching the layout exactly means
// a host-built pgl byte payload (via the Python `pgl.tensors_to_pgl`) can be
// passed as-is to a kernel accepting `megakittens::pgl_simple<GL, N>`.
//
// Layout (simplified from kittens::pgl — we only use per-peer gl access, no
// multicast TMA descriptor dict):
//   GL gls[N];
//   unsigned long long mc_size;       // unused, always 0
//   unsigned long long mc_handle;     // unused, always 0
//   GL::dtype *mc_vas[N];             // unused, always 0
//   int device_ids[N];
//
// This is exactly what the Python `pgl.tensors_to_pgl` packs. Kernels access
// `pgl.gls[peer_idx]` for per-device TMA via the underlying GL's own
// descriptor (embedded in the GL struct itself).

#include "kittens.cuh"

namespace megakittens {

template <typename GL, int NUM_DEVICES>
struct pgl_simple {
    GL gls[NUM_DEVICES];
    unsigned long long mc_size;
    unsigned long long mc_handle;
    typename GL::dtype *mc_vas[NUM_DEVICES];
    int device_ids[NUM_DEVICES];

    static constexpr int NUM_DEVS = NUM_DEVICES;

    __host__ __device__ inline const GL &operator[](int idx) const { return gls[idx]; }
};

} // namespace megakittens
