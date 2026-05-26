GPU := B200
SRC := kernels.cu
OUT := _C$(shell python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python -m examples.llama70b.manual_kernels.test_qkv_rope_append
CONFIG := pytorch

PYTHON_VERSION   := $(shell python -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION'))")
PYTHON_INCLUDES  := $(shell python -c "import sysconfig; print('-I' + sysconfig.get_path('include'))")
PYTHON_LIBDIR    := $(shell python -c "import sysconfig; print('-L' + sysconfig.get_config_var('LIBDIR'))")
PYBIND_INCLUDES  := $(shell python -m pybind11 --includes)
PYTORCH_INCLUDES := $(shell python -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")
PYTORCH_LIBDIR   := $(shell python -c "from torch.utils.cpp_extension import library_paths; print(' '.join(['-L' + p for p in library_paths()]))")

CUDA_INCLUDE_DIR ?= $(shell python -c "import os; p=os.environ.get('CUDA_HOME','/usr/local/cuda'); print(p + '/include')")
NVCCFLAGS += -I$(CUDA_INCLUDE_DIR)
NVCCFLAGS += -I$(CUDA_INCLUDE_DIR)/cccl

export NVCC_PREPEND_FLAGS := -I$(CUDA_INCLUDE_DIR) -L$(shell python -c "import os; print(os.environ.get('CUDA_HOME','/usr/local/cuda'))")/lib -L$(shell python -c "import os; print(os.environ.get('CUDA_HOME','/usr/local/cuda'))")/lib/stubs
export PATH := $(shell python -c "import os,sys; print(os.path.dirname(os.path.dirname(sys.executable)))")/nvvm/bin:$(PATH)

include ../../../csrc/ThunderKittens/kernels/common.mk
