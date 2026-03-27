from __future__ import annotations

import struct
from typing import Any, Sequence

import cuda.bindings.driver as cuda_driver
import torch  # TODO: completely remove torch dependency and purely rely on CUDA API

from .schema.device import Device
from .schema.dtype import DType
from .schema.tensor import TensorMeta
from .schema.instruction import Instruction, InstructionMeta
from .jit.c_utils import pack_args
from .jit.pykittens import gl
from .jit.cuda_utils import (
    get_kernel_from_cubin_module,
    get_sm_arch,
    initialize_cuda_context,
    launch_kernel,
    load_cubin_module,
    set_kernel_dynamic_smem,
    unload_cubin_module,
)
from .jit.nvrtc_jit import compile_source_to_cubin


def _pack_uint8s_to_int32s(values: tuple[int, ...], count: int, pad: int = 0) -> list[int]:
    """Pack *count* uint8 values into count/4 int32s (little-endian, padded with *pad*)."""
    padded = list(values) + [pad] * max(0, count - len(values))
    result: list[int] = []
    for i in range(0, count, 4):
        chunk = bytes(padded[i : i + 4])
        (val,) = struct.unpack("<i", chunk)
        result.append(val)
    return result


def _pack_instruction(inst: Instruction) -> list[int]:
    """Pack a single Instruction into a flat list of 32 int32 values."""
    inst_packed: list[int] = [0] * 32

    # 0 (0-3B): icode
    inst_packed[0] = inst.icode

    # 1-4 (4-19B): src_tensors (16 uint8 -> 4 int32)
    inst_packed[1:5] = _pack_uint8s_to_int32s(inst.src_tensors, Instruction.MAX_SRC_TENSORS)

    # 5-6 (20-27B): dst_tensors (8 uint8 -> 2 int32)
    inst_packed[5:7] = _pack_uint8s_to_int32s(inst.dst_tensors, Instruction.MAX_DST_TENSORS)

    # 7-20 (28-83B): indices (14 int32, zero-padded)
    indices = list(inst.indices) + [0] * max(0, Instruction.MAX_INDICES - len(inst.indices))
    inst_packed[7:21] = indices

    # 21-22 (84-91B): src_barriers (8 uint8 -> 2 int32)
    inst_packed[21:23] = _pack_uint8s_to_int32s(inst.src_barriers, Instruction.MAX_SRC_BARRIERS)

    # 23-30 (92-123B): src_barrier_targets (8 int32, zero-padded)
    targets = list(inst.src_barrier_targets) + [0] * max(
        0, Instruction.MAX_SRC_BARRIER_TARGETS - len(inst.src_barrier_targets)
    )
    inst_packed[23:31] = targets

    # 31 (124-127B): dst_barrier (4 uint8 -> 1 int32, 0xFF means unused)
    inst_packed[31:32] = _pack_uint8s_to_int32s(inst.dst_barrier, Instruction.MAX_DST_BARRIERS, pad=0xFF)

    return inst_packed


def _pack_instructions(instructions: list[Instruction], *, device: str) -> torch.Tensor:
    """Pack a list of Instruction objects into an (N, 32) int32 tensor."""
    buf = [_pack_instruction(inst) for inst in instructions]
    return torch.tensor(buf, dtype=torch.int32, device=device)


def _validate_tensor_against_meta(
    tensor: torch.Tensor, meta: TensorMeta, label: str,
) -> None:
    torch_dtype = meta.dtype.torch_dtype
    if tensor.dtype != torch_dtype:
        raise RuntimeError(
            f"[MegaKittens] {label} dtype mismatch "
            f"(expected {torch_dtype}, got {tensor.dtype})"
        )
    elif tuple(tensor.shape) != tuple(meta.shape):
        raise RuntimeError(
            f"[MegaKittens] {label} shape mismatch "
            f"(expected {tuple(meta.shape)}, got {tuple(tensor.shape)})"
        )
    expected_device = str(meta.device)
    actual_device = str(tensor.device)
    if actual_device != expected_device:
        raise RuntimeError(
            f"[MegaKittens] {label} device mismatch "
            f"(expected {expected_device}, got {actual_device})"
        )


class Dispatcher:
    """
    Runtime dispatcher for a compiled MegaKernel plan.

    Stores pre-allocated workspace tensors plus packed int32 tensors for the
    instruction stream and barriers, ready for direct consumption by the CUDA
    MegaKernel.
    """

    # Must match default_config in csrc/schema.cuh
    INSTRUCTION_PIPE_STAGES = 2
    CLUSTER_SIZE = 2
    NUM_CONSUMER_WARPS = 8
    NUM_WARPS = 4 + NUM_CONSUMER_WARPS
    NUM_THREADS = NUM_WARPS * 32
    DYNAMIC_SEMAPHORES = 32
    PAGE_SIZE = 32768
    STATIC_SHARED_MEMORY_BASE = 512 + INSTRUCTION_PIPE_STAGES * (128 + 128 + DYNAMIC_SEMAPHORES*8)
    DYNAMIC_SHARED_MEMORY_ALIGN = 1024
    NUM_PAGES = (227*1024 - STATIC_SHARED_MEMORY_BASE - DYNAMIC_SHARED_MEMORY_ALIGN) // PAGE_SIZE
    DYNAMIC_SHARED_MEMORY = NUM_PAGES*PAGE_SIZE + DYNAMIC_SHARED_MEMORY_ALIGN

    def __init__(
        self,
        instruction_metas: list[InstructionMeta],
        tensor_metas: list[TensorMeta],
        instructions: list[Instruction],
        num_barriers: int,
        input_tensor_indices: Sequence[int],
        output_tensor_indices: Sequence[int],
        use_jit_cache: bool = True,
    ) -> None:
        if not tensor_metas:
            raise RuntimeError("[MegaKittens] 'tensor_metas' must not be empty")
        if not isinstance(tensor_metas, list) or not all(isinstance(t, TensorMeta) for t in tensor_metas):
            raise RuntimeError(
                "[MegaKittens] 'tensor_metas' must be a list of TensorMeta"
            )
        if not isinstance(instructions, list) or not all(isinstance(i, Instruction) for i in instructions):
            raise RuntimeError(
                "[MegaKittens] 'instructions' must be a list of Instruction"
            )
        if not isinstance(num_barriers, int) or num_barriers < 0:
            raise RuntimeError(
                f"[MegaKittens] 'num_barriers' must be a non-negative int, got {num_barriers!r}"
            )
        if not all(isinstance(i, int) and i >= 0 for i in input_tensor_indices):
            raise RuntimeError(
                "[MegaKittens] 'input_tensor_indices' must contain only non-negative ints"
            )
        if not all(isinstance(i, int) and i >= 0 for i in output_tensor_indices):
            raise RuntimeError(
                "[MegaKittens] 'output_tensor_indices' must contain only non-negative ints"
            )
        num_tensors = len(tensor_metas)
        for idx in input_tensor_indices:
            if idx >= num_tensors:
                raise RuntimeError(
                    f"[MegaKittens] input_tensor_index {idx} out of range [0, {num_tensors})"
                )
        for idx in output_tensor_indices:
            if idx >= num_tensors:
                raise RuntimeError(
                    f"[MegaKittens] output_tensor_index {idx} out of range [0, {num_tensors})"
                )
        devices = {str(m.device) for m in tensor_metas}
        if len(devices) > 1:
            raise RuntimeError(
                f"[MegaKittens] All tensor_metas must share the same device, got {devices}"
            )

        self.instruction_metas = instruction_metas
        self.device: Device = tensor_metas[0].device  # TODO: handle multi-GPU case
        self.tensor_metas: list[TensorMeta] = tensor_metas
        self.tensors: list[torch.Tensor | None] = [None] * len(tensor_metas)
        self._materialized: bool = False
        self.instructions: list[Instruction] = instructions
        self.instruction_tensor: torch.Tensor | None = None
        self.num_barriers: int = num_barriers
        self.barrier_tensor: torch.Tensor | None = None
        self.input_tensor_indices: tuple[int, ...] = tuple(input_tensor_indices)
        self._input_indices_set: frozenset[int] = frozenset(input_tensor_indices)
        self.output_tensor_indices: tuple[int, ...] = tuple(output_tensor_indices)
        self._kernel_fn: cuda_driver.CUfunction | None = None
        self._cubin_module: cuda_driver.CUmodule | None = None
        self.all_tensors: list[torch.Tensor | None] = [None] * (2 + len(tensor_metas))
        self.gls: list[gl | None] = [None] * len(self.all_tensors)
        self.use_jit_cache = use_jit_cache

    def __del__(self) -> None:
        if self._cubin_module is not None:
            unload_cubin_module(self._cubin_module)
            self._cubin_module = None
            self._kernel_fn = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)

    def call(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise RuntimeError("[MegaKittens] Dispatcher does not support keyword arguments")

        if len(args) != len(self.input_tensor_indices):
            raise RuntimeError(
                f"[MegaKittens] Dispatcher input count mismatch: expected {len(self.input_tensor_indices)} "
                f"but got {len(args)}"
            )

        if not self._materialized:
            self._materialize(args)
        else:
            self._materialize_inputs(args)

        self._launch()

        outputs = tuple(self.tensors[idx] for idx in self.output_tensor_indices)
        return outputs[0] if len(outputs) == 1 else outputs

    def _materialize(self, args: tuple[Any, ...]) -> None:
        """For the first call: validate & assign inputs, allocate non-input tensors."""
        # Assign input tensor references
        self._materialize_inputs(args)

        # Allocate non-input tensors
        for slot_idx, meta in enumerate(self.tensor_metas):
            if slot_idx in self._input_indices_set:
                continue
            self.tensors[slot_idx] = torch.empty(
                meta.shape, dtype=meta.dtype.torch_dtype, device=str(meta.device),
            )

        # Allocate instruction and barrier tensors
        self.instruction_tensor = _pack_instructions(self.instructions, device=str(self.device))
        self.barrier_tensor = torch.zeros(
            max(self.num_barriers, 1), dtype=torch.int32, device=str(self.device),
        )

        # Collect TMA types per tensor index from instruction metas
        tensor_tma_types: dict[int, list[st | sv]] = {}
        for inst_meta in self.instruction_metas:
            tensor_specs = list(zip(inst_meta.src_tensors, inst_meta.itype.inputs)) + \
                           list(zip(inst_meta.dst_tensors, inst_meta.itype.outputs))
            for tensor_idx, tensor_spec in tensor_specs:
                for tma_type in tensor_spec.tma_types:
                    tma_list = tensor_tma_types.setdefault(tensor_idx, [])
                    if tma_type not in tma_list:
                        tma_list.append(tma_type)

        # Build gls
        self.all_tensors = [self.instruction_tensor, self.barrier_tensor] + self.tensors
        for i, t in enumerate(self.all_tensors):
            mk_dtype = DType.from_torch(t.dtype)
            if i < 2:  # instructions and barriers
                self.gls[i] = gl(dtype=mk_dtype, b=1, d=1, r=-1, c=-1)
            else:
                tma = tensor_tma_types.get(i - 2, [])
                self.gls[i] = gl(dtype=mk_dtype, b=-1, d=-1, r=-1, c=-1, tma_types=tma)

        self._materialized = True

    def _materialize_inputs(self, args: tuple[Any, ...]) -> None:
        """Validate & assign input references only."""
        for input_arg_idx, tensor_idx in enumerate(self.input_tensor_indices):
            src = args[input_arg_idx]
            if not isinstance(src, torch.Tensor):
                raise RuntimeError(
                    f"[MegaKittens] Input {input_arg_idx} is not a torch.Tensor "
                    f"(type={type(src).__name__})"
                )
            _validate_tensor_against_meta(
                src, self.tensor_metas[tensor_idx], f"Input {input_arg_idx}"
            )
            self.tensors[tensor_idx] = src
            self.all_tensors[2 + tensor_idx] = src  # TODO: if input dims change, all other dims + gls should change

    def _compile_kernel(self) -> None:
        device_index = self.device.index if self.device.index else torch.cuda.current_device()  # TODO: handle multi-GPU case
        initialize_cuda_context(device_index)
        major, minor = get_sm_arch(device_index)  # TODO: Generate config and globals correctly based on instructions

        itype_includes = "\n".join(f'#include "{inst_meta.itype.cpp_include}"' for inst_meta in self.instruction_metas if inst_meta.itype.cpp_include)
        gl_fields = "\n".join(f"{self.gls[i + 2].cpp_type} tensor_{i};" for i in range(len(self.gls) - 2))
        gls_body = "\n".join(f"{'if' if i == 0 else 'else if'} constexpr (I == {i}) return tensor_{i};" for i in range(len(self.gls) - 2))
        dispatch_cases = []
        for inst_meta in self.instruction_metas:
            template = inst_meta.itype.cpp_template
            if template is None:
                raise RuntimeError(f"[MegaKittens] IType '{inst_meta.itype.name}' has no cpp_template")
            tensor_args = ",".join(str(t) for t in inst_meta.src_tensors + inst_meta.dst_tensors)
            op = template.format(tensors=tensor_args)
            dispatch_cases.append(f"case {inst_meta.icode}: return dispatch_instruction<{op}, worker_type, T>(args...);")
        dispatch_cases = "\n".join(dispatch_cases)
        source = f"""
            #include "megakittens.cuh"
            {itype_includes}
            namespace megakittens {{
                struct MKConfig : default_config {{}};
                struct MKGlobals {{
                    {self.gls[0].cpp_type} instructions;
                    {self.gls[1].cpp_type} barriers;
                    {gl_fields}
                    template <int I> __device__ __forceinline__ auto& gls() const {{{gls_body}}}
                }};
                template <WorkerType worker_type, typename T, typename Config, typename Globals, typename... Args>
                __device__ __forceinline__ static T dispatch_instruction(const int icode, Args &...args) {{
                    switch (icode) {{
                        {dispatch_cases}
                        default: asm volatile("{{trap;\\n}}");
                    }}
                }}
            }}
        """
        cubin, (kernel_name,) = compile_source_to_cubin(
            source, (b"megakittens::kernel<megakittens::MKConfig, megakittens::MKGlobals>",), major, minor,
            use_file_cache=self.use_jit_cache,
        )
        self._cubin_module = load_cubin_module(cubin)
        self._kernel_fn = get_kernel_from_cubin_module(self._cubin_module, kernel_name)
        set_kernel_dynamic_smem(self._kernel_fn, self.DYNAMIC_SHARED_MEMORY)

    def _launch(self) -> None:
        if self._kernel_fn is None:
            self._compile_kernel()
        device_index = self.device.index if self.device.index else torch.cuda.current_device()
        # Reset barriers before each launch
        if self.num_barriers > 0:
            self.barrier_tensor.zero_()
        _globals_holder, globals_packed = pack_args(
            [(g.tensor_to_gl(t), g.size, g.align) for g, t in zip(self.gls, self.all_tensors)]
        )
        stream = torch.cuda.current_stream(device_index).cuda_stream
        launch_kernel(
            self._kernel_fn,
            globals_packed,
            grid=(-(-len(self.instructions) // self.CLUSTER_SIZE) * self.CLUSTER_SIZE,),  # round up to CLUSTER_SIZE
            block=(self.NUM_THREADS,),
            dynamic_smem_bytes=self.DYNAMIC_SHARED_MEMORY,
            stream=stream,
            cluster=(self.CLUSTER_SIZE,),
        )
