# GPU configuration
GPU := B300

# Compiler configuration
NVCC := nvcc

# ThunderKittens configuration
THUNDERKITTENS_ROOT := ./csrc/ThunderKittens

# Python/PyTorch binding configuration
PYTHON_VERSION ?= $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION'))")
PYTHON_INCLUDES ?= $(shell python3 -c "import sysconfig; print('-I', sysconfig.get_path('include'), sep='')")
PYTHON_LIBDIR ?= $(shell python3 -c "import sysconfig; print('-L', sysconfig.get_config_var('LIBDIR'), sep='')")
PYBIND_INCLUDES ?= $(shell python3 -m pybind11 --includes)
PYTORCH_INCLUDES ?= $(shell python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))") # recommended: define as an environment variable in advance
PYTORCH_LIBDIR ?= $(shell python3 -c "from torch.utils.cpp_extension import library_paths; print(' '.join(['-L' + p for p in library_paths()]))") # recommended: define as an environment variable in advance

# NVCC flags
NVCCFLAGS := -DNDEBUG -lineinfo
NVCCFLAGS += --expt-extended-lambda --expt-relaxed-constexpr 
NVCCFLAGS += -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing 
NVCCFLAGS += -forward-unknown-to-host-compiler -ftemplate-backtrace-limit=0
NVCCFLAGS += -std=c++20 -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static
NVCCFLAGS += -O3 --use_fast_math
NVCCFLAGS += -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills 
NVCCFLAGS += -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype
NVCCFLAGS += -DKITTENS_BLACKWELL
NVCCFLAGS += -shared -fPIC
NVCCFLAGS += -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
NVCCFLAGS += -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
NVCCFLAGS += -DTORCH_API_INCLUDE_EXTENSION_H
NVCCFLAGS += -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=1
NVCCFLAGS += $(PYTHON_INCLUDES) $(PYTORCH_INCLUDES)
NVCCFLAGS += ${PYTHON_LIBDIR} ${PYTORCH_LIBDIR} -lpython${PYTHON_VERSION}
NVCCFLAGS += -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10
NVCCFLAGS += -diag-suppress 3189

# Architecture-specific flags
ifeq ($(GPU),B300)
NVCCFLAGS += -DKITTENS_BLACKWELL -gencode arch=compute_103a,code=sm_103a
else ifeq ($(GPU),B200)
NVCCFLAGS += -DKITTENS_BLACKWELL -gencode arch=compute_100a,code=sm_100a
else
$(error Unsupported GPU: $(GPU). Please set GPU to B200 or B300.)
endif

# Targets
OUT := ./megakittens/_C.abi3.so
OBJDIR := ./build
SRCDIR := ./csrc

# Source files
CU_SRCS := $(shell find $(SRCDIR) -name '*.cu')
CPP_SRCS := $(shell find $(SRCDIR) -name '*.cpp')
SRCS := $(CU_SRCS) $(CPP_SRCS)

# Object files
CU_OBJS := $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.cu.o,$(CU_SRCS))
CPP_OBJS := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.cpp.o,$(CPP_SRCS))
OBJS := $(CU_OBJS) $(CPP_OBJS)

# Dependency files
DEPS := $(OBJS:.o=.d)

all: $(OUT)

# TODO: create unit tests
# test: $(OUT)

$(OUT): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@

$(OBJDIR)/%.cu.o: $(SRCDIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) -c $< $(NVCCFLAGS) -MMD -MP -MF $(@:.o=.d) -MT $@ -o $@

$(OBJDIR)/%.cpp.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) -c $< $(NVCCFLAGS) -MMD -MP -MF $(@:.o=.d) -MT $@ -o $@

-include $(DEPS)

clean:
	rm -rf $(OBJDIR)
	rm -f $(OUT)

.PHONY: all clean
