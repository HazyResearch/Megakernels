# Compiler
NVCC?=nvcc

ifndef GPU
GPU=B200
endif

TARGET=mk_llama
SRC=llama.cu

# Python version configuration
ifndef PYTHON_VERSION
PYTHON_VERSION=3.13
endif

NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lineinfo
NVCCFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${MEGAKERNELS_ROOT}/include $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags) -shared -Xcompiler=-fPIC -lpython${PYTHON_VERSION}
ifneq ($(CUDA_ROOT),)
NVCCFLAGS+= -I$(CUDA_ROOT)/include/cccl -L$(CUDA_ROOT)/lib
endif

# Conditional setup based on the target GPU
ifeq ($(GPU),4090)
NVCCFLAGS+= -DKITTENS_4090 -arch=sm_89
else ifeq ($(GPU),RTX5090)
# KITTENS_5090 selects the consumer shared-memory/register path in the VM;
# KITTENS_SM120 selects ThunderKittens' native Blackwell consumer path.
NVCCFLAGS+= -DKITTENS_SM120 -DKITTENS_5090 -arch=sm_120
else ifeq ($(GPU),A100)
NVCCFLAGS+= -DKITTENS_A100 -arch=sm_80
else ifeq ($(GPU),H100)
NVCCFLAGS+= -DKITTENS_HOPPER -arch=sm_90a
else
NVCCFLAGS+= -DKITTENS_HOPPER -DKITTENS_BLACKWELL -arch=sm_100a
endif

# Default target
all: $(TARGET)

# run: $(TARGET)
# 	python test.py

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)$(shell python3-config --extension-suffix)

# Clean target
clean:
	rm -f $(TARGET)$(shell python3-config --extension-suffix)
