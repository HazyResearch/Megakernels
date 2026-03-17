import functools
import json
import os
from pathlib import Path

import cuda.bindings.driver as cuda_driver


def check_cuda(err: cuda_driver.CUresult) -> None:
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error: {err}")


@functools.cache
def initialize_cuda_context(device_index: int = 0) -> None:
    (err,) = cuda_driver.cuInit(0)
    check_cuda(err)
    err, dev = cuda_driver.cuDeviceGet(device_index)
    check_cuda(err)
    err, ctx = cuda_driver.cuDevicePrimaryCtxRetain(dev)
    check_cuda(err)
    (err,) = cuda_driver.cuCtxSetCurrent(ctx)
    check_cuda(err)


@functools.cache
def get_sm_arch(device_index: int = 0) -> tuple[int, int]:
    err, dev = cuda_driver.cuDeviceGet(device_index)
    check_cuda(err)
    err, major = cuda_driver.cuDeviceGetAttribute(
        cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev
    )
    check_cuda(err)
    err, minor = cuda_driver.cuDeviceGetAttribute(
        cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev
    )
    check_cuda(err)
    return major, minor


@functools.cache
def get_cuda_driver_version() -> int:
    err, version = cuda_driver.cuDriverGetVersion()
    check_cuda(err)
    return version


@functools.cache
def cuda_include_dirs() -> tuple[str, ...]:
    def _check(base: Path) -> Path | None:
        p = base / "include"
        if (p / "cuda_bf16.h").exists():
            return p
        return None

    # 1. Explicit env vars
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        env_val = os.environ.get(env_var)
        if env_val and (cuda_include := _check(Path(env_val))):
            break
    else:
        # 2. Infer from PATH / LD_LIBRARY_PATH
        for env_var in ("PATH", "LD_LIBRARY_PATH"):
            env_val = os.environ.get(env_var, "")
            for entry in env_val.split(os.pathsep):
                if not entry:
                    continue
                candidate = Path(entry).parent  # remove bin/ or include/
                if cuda_include := _check(candidate):
                    break
            else:
                continue
            break
        else:
            # 3. Common system paths
            for base in ("/usr/local/cuda", "/usr/cuda"):
                if cuda_include := _check(Path(base)):
                    break
            else:
                raise RuntimeError("Cannot find CUDA include directory")

    # Verify include dir matches the running CUDA driver version
    version_file = cuda_include.parent / "version.json"
    if version_file.exists():
        ver_parts = json.loads(version_file.read_text())["cuda"]["version"].split(".")
        toolkit_major, toolkit_minor = int(ver_parts[0]), int(ver_parts[1])
        driver_ver = get_cuda_driver_version()
        driver_major, driver_minor = driver_ver // 1000, (driver_ver % 1000) // 10
        if (toolkit_major, toolkit_minor) != (driver_major, driver_minor):
            raise RuntimeError(
                f"[MegaKittens] CUDA toolkit version ({toolkit_major}.{toolkit_minor}) does not match "
                f"currently used driver version ({driver_major}.{driver_minor}). "
                f"Please ensure your CUDA toolkit installation is correct and set "
                f"CUDA_HOME, CUDA_PATH, PATH, or LD_LIBRARY to the correct paths."
            )

    cccl = cuda_include / "cccl"
    if cccl.exists():  # CUDA 13
        return (str(cuda_include), str(cccl), str(cccl / "cuda" / "std"))
    else:  # CUDA 12
        return (str(cuda_include), str(cuda_include / "cuda" / "std"))


def load_cubin_module(cubin: bytes) -> cuda_driver.CUmodule:
    err, module = cuda_driver.cuModuleLoadData(cubin)
    check_cuda(err)
    return module


def get_kernel_from_cubin_module(
    module: cuda_driver.CUmodule, kernel_name: bytes
) -> cuda_driver.CUfunction:
    err, fn = cuda_driver.cuModuleGetFunction(module, kernel_name)
    check_cuda(err)
    return fn


def unload_cubin_module(module: cuda_driver.CUmodule) -> None:
    (err,) = cuda_driver.cuModuleUnload(module)
    check_cuda(err)


def set_kernel_dynamic_smem(fn: cuda_driver.CUfunction, dynamic_smem_bytes: int) -> None:
    (err,) = cuda_driver.cuFuncSetAttribute(
        fn,
        cuda_driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        dynamic_smem_bytes
    )
    check_cuda(err)
