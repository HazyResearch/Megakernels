GPU := B200
SRC := kernels.cu
OUT := _C$(shell python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python -m examples.llama70b.manual_kernels.test_rms
CONFIG := pytorch

# Pin python lookups to the active conda env (use `python`, not `python3`,
# because `python3` may resolve to a different env on this machine).
PYTHON_VERSION   := $(shell python -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION'))")
PYTHON_INCLUDES  := $(shell python -c "import sysconfig; print('-I' + sysconfig.get_path('include'))")
PYTHON_LIBDIR    := $(shell python -c "import sysconfig; print('-L' + sysconfig.get_config_var('LIBDIR'))")
PYBIND_INCLUDES  := $(shell python -m pybind11 --includes)
PYTORCH_INCLUDES := $(shell python -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")
PYTORCH_LIBDIR   := $(shell python -c "from torch.utils.cpp_extension import library_paths; print(' '.join(['-L' + p for p in library_paths()]))")

# Make sure nvcc can find cuda_runtime.h. CUDA_HOME is exported by the conda env
# (e.g. /root/miniconda3/envs/nash/targets/x86_64-linux) but nvcc 13.0 doesn't
# always pick its toolkit-relative include dir up on its own.
CUDA_INCLUDE_DIR ?= $(shell python -c "import os; p=os.environ.get('CUDA_HOME','/usr/local/cuda'); print(p + '/include')")
NVCCFLAGS += -I$(CUDA_INCLUDE_DIR)
# CUDA 13 moved thrust under cccl/; add it so torch's `#include <thrust/...>` resolves.
NVCCFLAGS += -I$(CUDA_INCLUDE_DIR)/cccl

# Match the working TK kernel build env: prepend CUDA include/lib and put
# nvvm/bin (cicc) on PATH so nvcc can find its device-side compiler.
export NVCC_PREPEND_FLAGS := -I$(CUDA_INCLUDE_DIR) -L$(shell python -c "import os; print(os.environ.get('CUDA_HOME','/usr/local/cuda'))")/lib -L$(shell python -c "import os; print(os.environ.get('CUDA_HOME','/usr/local/cuda'))")/lib/stubs
export PATH := $(shell python -c "import os,sys; print(os.path.dirname(os.path.dirname(sys.executable)))")/nvvm/bin:$(PATH)

include ../../../csrc/ThunderKittens/kernels/common.mk
