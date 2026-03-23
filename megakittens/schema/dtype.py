from enum import Enum

import cuda.bindings.driver as cuda_driver
import torch


class DType(str, Enum):
    """MegaKittens dtype with properties for torch, C++, and TMA interop."""
    int32 = "int32"
    int16 = "int16"
    int8 = "int8"
    fp64 = "fp64"
    fp32 = "fp32"
    bf16 = "bf16"
    half = "half"
    fp8e4m3 = "fp8e4m3"
    fp8e5m2 = "fp8e5m2"
    fp8e8m0 = "fp8e8m0"
    fp4e2m1x2 = "fp4e2m1x2"

    @property
    def size(self) -> int:
        return DTYPE_SIZE[self]

    @property
    def torch_dtype(self) -> torch.dtype:
        return MK_TO_TORCH_DTYPE[self]

    @property
    def cpp_dtype(self) -> str:
        return MK_TO_CPP_DTYPE[self]

    @property
    def tma_dtype(self) -> cuda_driver.CUtensorMapDataType:
        return MK_TO_TMA_DTYPE[self]

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> "DType":
        if dtype not in TORCH_TO_MK_DTYPE:
            raise ValueError(f"[MegaKittens] Unsupported torch dtype: {dtype}")
        return TORCH_TO_MK_DTYPE[dtype]


DTYPE_SIZE: dict[DType, int] = {
    DType.int32: 4,
    DType.int16: 2,
    DType.int8: 1,
    DType.fp64: 8,
    DType.fp32: 4,
    DType.bf16: 2,
    DType.half: 2,
    DType.fp8e4m3: 1,
    DType.fp8e5m2: 1,
    DType.fp8e8m0: 1,
    DType.fp4e2m1x2: 1,
}

MK_TO_TORCH_DTYPE: dict[DType, torch.dtype] = {
    DType.fp64: torch.float64,
    DType.fp32: torch.float32,
    DType.bf16: torch.bfloat16,
    DType.half: torch.float16,
    DType.fp8e4m3: torch.float8_e4m3fn,
    DType.fp8e5m2: torch.float8_e5m2fnuz,
    DType.fp8e8m0: torch.float8_e8m0fnu,
    DType.fp4e2m1x2: torch.float4_e2m1fn_x2,
    DType.int32: torch.int32,
    DType.int16: torch.int16,
    DType.int8: torch.int8,
}

TORCH_TO_MK_DTYPE: dict[torch.dtype, DType] = {v: k for k, v in MK_TO_TORCH_DTYPE.items()}

MK_TO_CPP_DTYPE: dict[DType, str] = {
    DType.int32: "int",
    DType.int16: "short",
    DType.int8: "int8_t",
    DType.fp64: "double",
    DType.fp32: "float",
    DType.bf16: "kittens::bf16",
    DType.half: "kittens::half",
    DType.fp8e4m3: "kittens::fp8e4m3",
    DType.fp8e5m2: "kittens::fp8e5m2",
    DType.fp8e8m0: "kittens::fp8e8m0",
    DType.fp4e2m1x2: "kittens::fp4e2m1_2",
}

MK_TO_TMA_DTYPE: dict[DType, cuda_driver.CUtensorMapDataType] = {
    DType.fp32: cuda_driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    DType.half: cuda_driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    DType.bf16: cuda_driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    DType.fp8e4m3: cuda_driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    DType.fp8e8m0: cuda_driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    DType.fp4e2m1x2: cuda_driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
}
