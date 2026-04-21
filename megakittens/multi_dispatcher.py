"""Multi-device PGL-aware dispatcher.

TODO: consolidate with Dispatcher once the port stabilizes.

Bridges the framework's single-device Dispatcher pipeline to an N-device launch
where one tensor slot can be a PGL (replicated across devices, peer-accessible
via NVLink under P2P). This is deliberately NOT integrated with torch.compile
— the user constructs it directly from IType + per-device input tensors.

Scope: enough framework integration to run DownProjResidual on 8×B200 end-to-end
through the ThunderKittens megakernel runtime (instruction stream, barriers,
controller/loader/launcher/consumer/storer workers, tcgen05 cluster MMA). The
torch.compile FX-graph-aware sharding path is a separate (much larger) project.
"""

from __future__ import annotations

from typing import Sequence

import cuda.bindings.driver as cuda_driver
import torch

from .dispatcher import Dispatcher, _pack_instructions_global
from .schema.dtype import DType
from .schema.tensor import TensorMeta
from .schema.instruction import Instruction, InstructionMeta
from .jit.c_utils import c_int, pack_args
from .jit.cuda_utils import (
    check_cuda,
    get_kernel_from_cubin_module,
    get_sm_arch,
    initialize_cuda_context,
    launch_kernel,
    load_cubin_module,
    set_kernel_dynamic_smem,
    unload_cubin_module,
)
from .jit.nvrtc_jit import compile_source_to_cubin
from .jit.pykittens import gl, pgl


def _enable_all_p2p(num_devices: int) -> list:
    """Retain primary contexts for all devices and pairwise-enable P2P access."""
    ctxs = []
    for i in range(num_devices):
        initialize_cuda_context(i)
        err, dev = cuda_driver.cuDeviceGet(i)
        check_cuda(err)
        err, ctx = cuda_driver.cuDevicePrimaryCtxRetain(dev)
        check_cuda(err)
        ctxs.append(ctx)
    for src in range(num_devices):
        (err,) = cuda_driver.cuCtxSetCurrent(ctxs[src])
        check_cuda(err)
        for dst in range(num_devices):
            if src == dst:
                continue
            (err,) = cuda_driver.cuCtxEnablePeerAccess(ctxs[dst], 0)
            if err not in (
                cuda_driver.CUresult.CUDA_SUCCESS,
                cuda_driver.CUresult.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
            ):
                raise RuntimeError(f"cuCtxEnablePeerAccess({src}->{dst}) failed: {err}")
    return ctxs


class MultiDispatcher:
    """PGL-aware multi-device launcher.

    Takes the same schedule output as Dispatcher plus ``num_devices`` and
    ``pgl_tensor_indices``. Each pgl slot's tensor exists on every device (same
    shape/dtype); on launch the packed globals contain all N peer pointers so
    any device can TMA into any peer's slot.
    """

    CLUSTER_SIZE = 2

    def __init__(
        self,
        instruction_metas: list[InstructionMeta],
        tensor_metas: list[TensorMeta],
        instructions: list[Instruction],
        num_barriers: int,
        input_tensor_indices: Sequence[int],
        output_tensor_indices: Sequence[int],
        num_devices: int,
        pgl_tensor_indices: Sequence[int] = (),
        use_jit_cache: bool = False,
    ) -> None:
        if num_devices < 2:
            raise RuntimeError(f"[MegaKittens] MultiDispatcher requires num_devices >= 2, got {num_devices}")
        self.num_devices = num_devices
        self.instruction_metas = instruction_metas
        self.tensor_metas = tensor_metas
        self.instructions = instructions
        self.num_barriers = num_barriers
        self.input_tensor_indices = tuple(input_tensor_indices)
        self._input_indices_set = frozenset(input_tensor_indices)
        self.output_tensor_indices = tuple(output_tensor_indices)
        self.pgl_tensor_indices = frozenset(pgl_tensor_indices)
        self.use_jit_cache = use_jit_cache

        # Per-device state — populated on first call.
        self._ctxs = _enable_all_p2p(num_devices)
        # tensors_per_dev[dev][slot] = torch.Tensor
        self.tensors_per_dev: list[list[torch.Tensor | None]] = [
            [None] * len(tensor_metas) for _ in range(num_devices)
        ]
        self.instruction_tensors: list[torch.Tensor | None] = [None] * num_devices
        self.barrier_tensors: list[torch.Tensor | None] = [None] * num_devices
        self._materialized = False
        self._cubin: bytes | None = None
        self._mangled: bytes | None = None
        self._modules_per_dev: list = [None] * num_devices
        self._kernel_fn_per_dev: list = [None] * num_devices
        self._packed_args_per_dev: list = [None] * num_devices

    def __call__(self, *args_per_dev: list[Sequence[torch.Tensor]]) -> list[list[torch.Tensor]]:
        """Call with per-device input tensors.

        args_per_dev[dev_idx] = tuple of input tensors for that device, in
        input_tensor_indices order.

        Returns outputs[slot_idx] = list[torch.Tensor] with length num_devices.
        """
        if len(args_per_dev) != self.num_devices:
            raise RuntimeError(
                f"[MegaKittens] MultiDispatcher expects {self.num_devices} positional args (one per device), "
                f"got {len(args_per_dev)}"
            )
        if not self._materialized:
            self._materialize(args_per_dev)
        else:
            self._materialize_inputs(args_per_dev)

        self._launch()

        outputs: list[list[torch.Tensor]] = []
        for slot in self.output_tensor_indices:
            outputs.append([self.tensors_per_dev[dev][slot] for dev in range(self.num_devices)])
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def _materialize_inputs(self, args_per_dev) -> None:
        for dev in range(self.num_devices):
            args = args_per_dev[dev]
            if len(args) != len(self.input_tensor_indices):
                raise RuntimeError(
                    f"[MegaKittens] dev {dev} expected {len(self.input_tensor_indices)} inputs, got {len(args)}"
                )
            for input_arg_idx, tensor_idx in enumerate(self.input_tensor_indices):
                src = args[input_arg_idx]
                if not isinstance(src, torch.Tensor):
                    raise RuntimeError(
                        f"[MegaKittens] dev {dev} input {input_arg_idx} is not a torch.Tensor"
                    )
                self.tensors_per_dev[dev][tensor_idx] = src
                if str(src.device) != f"cuda:{dev}":
                    raise RuntimeError(
                        f"[MegaKittens] dev {dev} input {input_arg_idx} on wrong device: {src.device}"
                    )

    def _materialize(self, args_per_dev) -> None:
        self._materialize_inputs(args_per_dev)

        for dev in range(self.num_devices):
            for slot, meta in enumerate(self.tensor_metas):
                if slot in self._input_indices_set:
                    continue
                with torch.cuda.device(dev):
                    self.tensors_per_dev[dev][slot] = torch.zeros(
                        meta.shape, dtype=meta.dtype.torch_dtype, device=f"cuda:{dev}"
                    )

        for dev in range(self.num_devices):
            with torch.cuda.device(dev):
                self.instruction_tensors[dev] = _pack_instructions_global(
                    self.instructions, device=f"cuda:{dev}"
                )
                self.barrier_tensors[dev] = torch.zeros(
                    max(self.num_barriers, 1), dtype=torch.int32, device=f"cuda:{dev}"
                )

        tensor_tma_types: dict[int, list] = {}
        for inst_meta in self.instruction_metas:
            specs = list(zip(inst_meta.src_tensors, inst_meta.itype.inputs)) + list(
                zip(inst_meta.dst_tensors, inst_meta.itype.outputs)
            )
            for tensor_idx, tensor_spec in specs:
                for tma_type in tensor_spec.tma_types:
                    tma_list = tensor_tma_types.setdefault(tensor_idx, [])
                    if tma_type not in tma_list:
                        tma_list.append(tma_type)

        self.all_slots_descriptors: list[gl | pgl] = [
            gl(dtype=DType.from_torch(self.instruction_tensors[0].dtype), b=1, d=1, r=-1, c=-1),
            gl(dtype=DType.from_torch(self.barrier_tensors[0].dtype), b=1, d=1, r=-1, c=-1),
        ]
        for slot_idx in range(len(self.tensor_metas)):
            mk_dtype = DType.from_torch(self.tensors_per_dev[0][slot_idx].dtype)
            inner = gl(dtype=mk_dtype, b=-1, d=-1, r=-1, c=-1,
                       tma_types=tensor_tma_types.get(slot_idx, []))
            if slot_idx in self.pgl_tensor_indices:
                self.all_slots_descriptors.append(pgl(inner=inner, num_devices=self.num_devices))
            else:
                self.all_slots_descriptors.append(inner)

        self._compile_kernel()
        for dev in range(self.num_devices):
            self._build_packed_args_for_dev(dev)
        self._materialized = True

    def _compile_kernel(self) -> None:
        initialize_cuda_context(0)
        major, minor = get_sm_arch(0)

        itype_includes = "\n".join(
            f'#include "{im.itype.cpp_include}"' for im in self.instruction_metas if im.itype.cpp_include
        )

        gl_fields = "\n".join(
            f"{self.all_slots_descriptors[i + 2].cpp_type} tensor_{i};"
            for i in range(len(self.tensor_metas))
        )

        # gls<I>() returns regular gl slots; pgls<I>() returns pgl slots.
        gl_cases, pgl_cases = [], []
        for i in range(len(self.tensor_metas)):
            case = f"if constexpr (I == {i}) return tensor_{i};"
            (pgl_cases if i in self.pgl_tensor_indices else gl_cases).append(case)
        gls_body = "\n".join(gl_cases) or 'static_assert(I == -9999, "no gl slots");'
        pgls_body = "\n".join(pgl_cases) or 'static_assert(I == -9999, "no pgl slots");'

        dispatch_cases = []
        for im in self.instruction_metas:
            template = im.itype.cpp_template
            if template is None:
                raise RuntimeError(f"[MegaKittens] IType '{im.itype.name}' has no cpp_template")
            tensor_args = ",".join(str(t) for t in im.src_tensors + im.dst_tensors)
            op = template.format(tensors=tensor_args)
            dispatch_cases.append(
                f"case {im.icode}: return dispatch_instruction<{op}, worker_type, T>(args...);"
            )
        dispatch_cases_s = "\n".join(dispatch_cases)

        instr_cpp = self.all_slots_descriptors[0].cpp_type
        barr_cpp = self.all_slots_descriptors[1].cpp_type

        # Python-computed pgl size must match C++ sizeof, else packed globals
        # won't align with what the kernel reads.
        layout_asserts = "\n".join(
            f'static_assert(sizeof({self.all_slots_descriptors[i + 2].cpp_type}) == {self.all_slots_descriptors[i + 2].size}, '
            f'"pgl slot {i} size mismatch");'
            for i in self.pgl_tensor_indices
        )

        source = f"""
            #include "megakittens.cuh"
            {itype_includes}
            namespace megakittens {{
                {layout_asserts}
                struct MKConfig : default_config {{}};
                struct MKGlobals {{
                    static constexpr int NUM_DEVICES = {self.num_devices};
                    {instr_cpp} instructions;
                    {barr_cpp} barriers;
                    {gl_fields}
                    int dev_idx;
                    template <int I> __device__ __forceinline__ auto& gls() const {{{gls_body}}}
                    template <int I> __device__ __forceinline__ auto& pgls() const {{{pgls_body}}}
                }};
                template <WorkerType worker_type, typename T, typename Config, typename Globals, typename... Args>
                __device__ __forceinline__ static T dispatch_instruction(const int icode, Args &...args) {{
                    switch (icode) {{
                        {dispatch_cases_s}
                        default: asm volatile("{{trap;\\n}}");
                    }}
                }}
            }}
        """
        cubin, (kernel_name,) = compile_source_to_cubin(
            source,
            (b"megakittens::kernel<megakittens::MKConfig, megakittens::MKGlobals>",),
            major, minor, use_file_cache=self.use_jit_cache,
        )
        self._cubin = cubin
        self._mangled = kernel_name

        # Cubin modules are per-context, so load into each device separately.
        for dev in range(self.num_devices):
            with torch.cuda.device(dev):
                module = load_cubin_module(cubin)
                fn = get_kernel_from_cubin_module(module, kernel_name)
                set_kernel_dynamic_smem(fn, Dispatcher.DYNAMIC_SHARED_MEMORY)
                self._modules_per_dev[dev] = module
                self._kernel_fn_per_dev[dev] = fn

    def _build_packed_args_for_dev(self, dev: int) -> None:
        """Pack MKGlobals for one device. PGL slots get all N peer pointers
        (byte-identical across devices); gl slots get this device's tensor."""
        fields = []
        g0 = self.all_slots_descriptors[0]
        fields.append((g0.tensor_to_gl(self.instruction_tensors[dev]), g0.size, g0.align))
        g1 = self.all_slots_descriptors[1]
        fields.append((g1.tensor_to_gl(self.barrier_tensors[dev]), g1.size, g1.align))
        for slot_idx in range(len(self.tensor_metas)):
            desc = self.all_slots_descriptors[slot_idx + 2]
            if slot_idx in self.pgl_tensor_indices:
                per_device_ptrs = [self.tensors_per_dev[d][slot_idx].data_ptr() for d in range(self.num_devices)]
                per_device_shapes = []
                for d in range(self.num_devices):
                    t = self.tensors_per_dev[d][slot_idx]
                    shape = [1, 1, 1, 1]
                    for i in range(t.ndim):
                        shape[4 - t.ndim + i] = t.shape[i]
                    per_device_shapes.append(tuple(shape))
                fields.append((desc.tensors_to_pgl(per_device_ptrs, per_device_shapes), desc.size, desc.align))
            else:
                t = self.tensors_per_dev[dev][slot_idx]
                fields.append((desc.tensor_to_gl(t), desc.size, desc.align))
        fields.append((c_int(dev), 4, 4))  # dev_idx

        # `pack_args` returns a holder whose lifetime owns the packed buffer's storage.
        self._packed_args_per_dev[dev] = pack_args(fields)

    def _launch(self) -> None:
        # Input tensor pointers can change call-to-call, so re-pack every time.
        for dev in range(self.num_devices):
            with torch.cuda.device(dev):
                if self.num_barriers > 0:
                    self.barrier_tensors[dev].zero_()
            self._build_packed_args_for_dev(dev)

        grid_x = (-(-len(self.instructions) // self.CLUSTER_SIZE)) * self.CLUSTER_SIZE

        for dev in range(self.num_devices):
            with torch.cuda.device(dev):
                (_holder, packed) = self._packed_args_per_dev[dev]
                stream = torch.cuda.current_stream(dev).cuda_stream
                launch_kernel(
                    self._kernel_fn_per_dev[dev], packed,
                    grid=(grid_x,),
                    block=(Dispatcher.NUM_THREADS,),
                    dynamic_smem_bytes=Dispatcher.DYNAMIC_SHARED_MEMORY,
                    stream=stream,
                    cluster=(self.CLUSTER_SIZE,),
                )

        # Synchronize all devices.
        for dev in range(self.num_devices):
            with torch.cuda.device(dev):
                torch.cuda.synchronize()

    def __del__(self) -> None:
        for module in self._modules_per_dev:
            if module is not None:
                try:
                    unload_cubin_module(module)
                except Exception:
                    pass
