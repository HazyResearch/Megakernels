# Compiler
NVCC?=nvcc

TARGET=mk_llama_tp
SRC=llama.cu
PCH=pch.cuh
PCH_GCH=$(PCH).gch

# Python version configuration
ifndef PYTHON_VERSION
PYTHON_VERSION=3.13
endif

# GPU configuration
ifndef GPU
GPU=B200
endif

NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas -lineinfo
NVCCFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${MEGAKERNELS_ROOT}/include $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags) -shared -fPIC -lpython${PYTHON_VERSION}

# Precompiled header flags
PCH_FLAGS=-include $(PCH)

# Extension suffix
EXT_SUFFIX := $(shell python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

# Conditional setup based on the target GPU
ifeq ($(GPU),4090)
NVCCFLAGS+= -DKITTENS_4090 -arch=sm_89
else ifeq ($(GPU),A100)
NVCCFLAGS+= -DKITTENS_A100 -arch=sm_80
else ifeq ($(GPU),H100)
NVCCFLAGS+= -DKITTENS_HOPPER -arch=sm_90a
else
NVCCFLAGS+= -DKITTENS_HOPPER -DKITTENS_BLACKWELL -arch=sm_100a
endif

EXTRA_NVCCFLAGS ?=
NVCCFLAGS+= $(EXTRA_NVCCFLAGS)

# Default target
all: $(TARGET)

# run: $(TARGET)
# 	python test.py

# Precompiled header target
$(PCH_GCH): $(PCH)
	$(NVCC) $(PCH) $(NVCCFLAGS) -o $(PCH_GCH)

$(TARGET): $(PCH_GCH) $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) $(PCH_FLAGS) -o $(TARGET)$(EXT_SUFFIX)

# Clean target
clean:
	rm -f $(TARGET)$(EXT_SUFFIX) $(PCH_GCH)
