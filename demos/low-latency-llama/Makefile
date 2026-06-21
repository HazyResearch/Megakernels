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

NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas -lineinfo
NVCCFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${MEGAKERNELS_ROOT}/include $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags) -shared -fPIC -lpython${PYTHON_VERSION}

# Conditional setup based on the target GPU
ifeq ($(GPU),4090)
NVCCFLAGS+= -DKITTENS_4090 -arch=sm_89
else ifeq ($(GPU),A100)
NVCCFLAGS+= -DKITTENS_A100 -arch=sm_80
else ifeq ($(GPU),H100)
NVCCFLAGS+= -DKITTENS_HOPPER -arch=sm_90a
else ifeq ($(GPU),GB10)
# GB10 (DGX Spark, sm_121 consumer/Spark Blackwell). Build against a SM120-aware
# ThunderKittens (THUNDERKITTENS_ROOT): KITTENS_SM120 alone selects the arch and
# enables the Hopper-equivalent features the megakernel needs (TMA, setmaxnreg,
# warpgroup MMA via the SM120 wgmma shim). Do NOT also pass KITTENS_HOPPER -- TK
# expands that to KITTENS_SM90 and then errors on two arch macros. NOT
# KITTENS_BLACKWELL: sm_121 lacks the sm_100a tcgen05 tensor-memory engine, so
# the megakernel takes its non-Blackwell (#else) CUDA-core / wgmma paths.
NVCCFLAGS+= -DKITTENS_SM120 -gencode arch=compute_121a,code=sm_121a
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
