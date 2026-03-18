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


def _cache_key(src: str, kernel_symbols: tuple[bytes, ...], major: int, minor: int) -> str:
    sm_suffix = "a" if major >= 9 else ""
    cuda_ver = get_cuda_driver_version()
    symbols_str = b"_".join(kernel_symbols).decode()
    payload = f"cuda_{cuda_ver}\nsm_{major}{minor}{sm_suffix}\n{symbols_str}\n{src}".encode()
    return hashlib.sha256(payload).hexdigest()


def _load_from_cache(key: str) -> tuple[bytes, tuple[bytes, ...]] | None:
    cubin_path = CUBIN_CACHE_DIR / f"{key}.cubin"
    mangled_names_path = CUBIN_CACHE_DIR / f"{key}.names"
    if cubin_path.exists() and mangled_names_path.exists():
        return cubin_path.read_bytes(), tuple(mangled_names_path.read_bytes().split(b"\n"))
    return None


def _save_to_cache(key: str, cubin: bytes, mangled_names: tuple[bytes, ...]) -> None:
    CUBIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CUBIN_CACHE_DIR / f"{key}.cubin").write_bytes(cubin)
    (CUBIN_CACHE_DIR / f"{key}.names").write_bytes(b"\n".join(mangled_names))


@functools.cache
def compile_source_to_cubin(
    src: str, kernel_symbols: tuple[bytes, ...], major: int, minor: int,
    *, use_file_cache: bool = True,
) -> tuple[bytes, tuple[bytes, ...]]:
    # 0. Check file-backed cache
    cache_key = _cache_key(src, kernel_symbols, major, minor)
    if use_file_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            return cached

    # 1. Create NVRTC program instance
    err, nvrtc_prog = nvrtc.nvrtcCreateProgram(
        src.encode("utf-8"),  # CUDA source code
        b"kernel.cu",         # program name (for diagnostics)
        0,                    # number of inline headers
        None,                 # array of inline header sources
        None,                 # array of inline header include names
    )
    check_nvrtc(err)

    # 2. Register kernel symbol expressions
    for kernel_symbol in kernel_symbols:
        (err,) = nvrtc.nvrtcAddNameExpression(nvrtc_prog, kernel_symbol)
        check_nvrtc(err)

    # 3. Prepare compiler flags and compile
    if major not in THUNDERKITTENS_ARCH_DEFINES:
        raise RuntimeError(f"[MegaKittens] Unsupported GPU compute capability: sm_{major}x")
    sm_suffix = "a" if major >= 9 else ""
    opts = tuple(flag.encode("utf-8") for flag in COMMON_NVRTC_FLAGS) + (
        THUNDERKITTENS_ARCH_DEFINES[major].encode("utf-8"),
        f"--gpu-architecture=sm_{major}{minor}{sm_suffix}".encode("utf-8"),
    )
    (err_compile,) = nvrtc.nvrtcCompileProgram(nvrtc_prog, len(opts), opts)

    # 4. Print compiler logs and check compilation error
    err, log_size = nvrtc.nvrtcGetProgramLogSize(nvrtc_prog)
    check_nvrtc(err)
    log = b" " * log_size
    (err,) = nvrtc.nvrtcGetProgramLog(nvrtc_prog, log)
    check_nvrtc(err)
    decoded_log = log.decode(errors="ignore").strip()
    if decoded_log:
        print(decoded_log)
    check_nvrtc(err_compile)

    # 5. Get mangled names
    mangled_names = []
    for kernel_symbol in kernel_symbols:
        err, name = nvrtc.nvrtcGetLoweredName(nvrtc_prog, kernel_symbol)
        check_nvrtc(err)
        mangled_names.append(name)
    mangled_names = tuple(mangled_names)

    # 6. Retrieve the compiled CUBIN binary
    err, cubin_size = nvrtc.nvrtcGetCUBINSize(nvrtc_prog)
    check_nvrtc(err)
    if cubin_size == 0:
        raise RuntimeError("NVRTC returned no CUBIN")
    cubin = b" " * cubin_size
    (err,) = nvrtc.nvrtcGetCUBIN(nvrtc_prog, cubin)
    check_nvrtc(err)

    # 7. Destroy NVRTC program instance
    (err,) = nvrtc.nvrtcDestroyProgram(nvrtc_prog)
    check_nvrtc(err)

    # 8. Save to file-backed cache
    if use_file_cache:
        _save_to_cache(cache_key, cubin, mangled_names)

    return cubin, mangled_names
