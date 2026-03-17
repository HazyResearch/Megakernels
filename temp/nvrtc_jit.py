import functools
import hashlib
from pathlib import Path

import cuda.bindings.nvrtc as nvrtc

from cuda_utils import cuda_include_dirs, get_cuda_driver_version


CUBIN_CACHE_DIR = Path.home() / ".cache" / "megakittens" / "cubin"
THUNDERKITTENS_ROOT = Path(__file__).resolve().parent.parent / "csrc" / "ThunderKittens"
THUNDERKITTENS_ARCH_DEFINES = {
    10: "-DKITTENS_BLACKWELL", 9: "-DKITTENS_HOPPER", 8: "-DKITTENS_AMPERE"
}
COMMON_NVRTC_FLAGS = (
    "--std=c++20",
    "--use_fast_math",
    "-Xptxas=--verbose",
    "-Xptxas=--warn-on-spills",
    "-DNDEBUG",
    "-lineinfo",
    "-DKITTENS_NO_HOST",
    f"-I{THUNDERKITTENS_ROOT / 'include'}",
    f"-I{THUNDERKITTENS_ROOT / 'prototype'}",
    *(f"-I{d}" for d in cuda_include_dirs()),
)


def check_nvrtc(err: nvrtc.nvrtcResult) -> None:
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"NVRTC error: {err}")


def _cubin_cache_key(src: str, major: int, minor: int) -> str:
    sm_suffix = "a" if major >= 9 else ""
    cuda_ver = get_cuda_driver_version()
    payload = f"cuda_{cuda_ver}\nsm_{major}{minor}{sm_suffix}\n{src}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_cubin_from_cache(key: str) -> bytes | None:
    path = CUBIN_CACHE_DIR / f"{key}.cubin"
    if path.exists():
        return path.read_bytes()
    return None


def _save_cubin_to_cache(key: str, cubin: bytes) -> None:
    CUBIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CUBIN_CACHE_DIR / f"{key}.cubin").write_bytes(cubin)


@functools.cache
def compile_source_to_cubin(src: str, major: int, minor: int) -> bytes:
    # 0. Check file-backed cache
    cache_key = _cubin_cache_key(src, major, minor)
    cached = _load_cubin_from_cache(cache_key)
    if cached is not None:
        return cached

    # 1. Create NVRTC program instance
    err, prog = nvrtc.nvrtcCreateProgram(
        src.encode("utf-8"),  # CUDA source code
        b"kernel.cu",         # program name (for diagnostics)
        0,                    # number of inline headers
        None,                 # array of inline header sources
        None,                 # array of inline header include names
    )
    check_nvrtc(err)

    # 2. Prepare compiler flags and compile
    if major not in THUNDERKITTENS_ARCH_DEFINES:
        raise RuntimeError(f"[MegaKittens] Unsupported GPU compute capability: sm_{major}x")
    sm_suffix = "a" if major >= 9 else ""
    opts = tuple(flag.encode("utf-8") for flag in COMMON_NVRTC_FLAGS) + (
        THUNDERKITTENS_ARCH_DEFINES[major].encode("utf-8"),
        f"--gpu-architecture=sm_{major}{minor}{sm_suffix}".encode("utf-8"),
    )
    (err_compile,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)

    # 3. Print compiler logs and check compilation error
    err, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
    check_nvrtc(err)
    log = b" " * log_size
    (err,) = nvrtc.nvrtcGetProgramLog(prog, log)
    check_nvrtc(err)
    decoded_log = log.decode(errors="ignore").strip()
    if decoded_log:
        print(decoded_log)
    check_nvrtc(err_compile)

    # 4. Retrieve the compiled CUBIN binary
    err, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
    check_nvrtc(err)
    if cubin_size == 0:
        raise RuntimeError("NVRTC returned no CUBIN")
    cubin = b" " * cubin_size
    (err,) = nvrtc.nvrtcGetCUBIN(prog, cubin)
    check_nvrtc(err)

    # 5. Destroy NVRTC program instance
    (err,) = nvrtc.nvrtcDestroyProgram(prog)
    check_nvrtc(err)

    # 6. Save to file-backed cache
    _save_cubin_to_cache(cache_key, cubin)

    return cubin
