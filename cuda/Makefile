# Builds the mumax3 GPU kernels and generates the Go wrappers.
#
# Two backends are supported, selected with BACKEND:
#
#   BACKEND=cuda (default): compile each .cu to PTX text with nvcc, one per
#   compute capability in $CUDA_CC (default "50"). cuda2go embeds the PTX into
#   <name>_wrapper.go keyed by compute capability. This is the upstream NVIDIA
#   build.
#
#   BACKEND=hip: compile each .cu to ONE generic SPIR-V (amdgcnspirv) image
#   with "hipcc --genco --offload-arch=amdgcnspirv". This is the faithful
#   analog of the CUDA path's embedded PTX: amdgcnspirv is a forward-compatible
#   virtual ISA that the ROCm runtime finalizes (JITs) at module load for
#   whatever GPU is present, so a single embedded image runs on any supported
#   gfx arch with no per-arch matrix and no rebuild for new GPUs. cuda2go embeds
#   the image bytes (base64) into <name>_wrapper_hip.go and the runtime loader
#   hands it straight to hipModuleLoadData. CUDA_CC does NOT drive the HIP build;
#   it stays meaningful only for the CUDA/nvcc path.
#
# Examples:
#
#    make                                          # nvcc, CC 50
#    make CUDA_CC="50 60 70 80 86 90"              # nvcc, several CCs
#    make CUDA_HOME=/usr/local/cuda-12.6 CUDA_CC=90
#    make BACKEND=hip                              # hipcc, single amdgcnspirv image
#    make BACKEND=hip HIPCC=/opt/rocm/bin/hipcc
#
# Different CUDA versions support different compute capabilities, as shown in the list below. See https://stackoverflow.com/a/28933055.
# CUDA SDK 10.0 support for compute capability       30 32 35 37 50 52 53 60 61 62 70 72 75
# CUDA SDK 10.1 support for compute capability       30 32 35 37 50 52 53 60 61 62 70 72 75
# CUDA SDK 10.2 support for compute capability       30 32 35 37 50 52 53 60 61 62 70 72 75
# CUDA SDK 11.0 support for compute capability             35 37 50 52 53 60 61 62 70 72 75 80
# CUDA SDK 11.1-11.7 support for compute capability        35 37 50 52 53 60 61 62 70 72 75 80 86
# CUDA SDK 11.8 support for compute capability             35 37 50 52 53 60 61 62 70 72 75 80 86 87 89
# CUDA SDK 12.0+ support for compute capability                  50 52 53 60 61 62 70 72 75 80 86 87 89 90

SHELL = /bin/bash

BACKEND ?= cuda

CUDAFILES := $(wildcard *.cu)

ifeq ($(BACKEND),hip)

ifeq ($(HIPCC),)
	HIPCC=hipcc
endif

# --offload-arch=amdgcnspirv emits one generic SPIR-V image per kernel that the
# ROCm runtime finalizes for the present GPU at hipModuleLoadData time; no gfx
# arch list is needed (CUDA_CC is a CUDA/nvcc-path concept only).
HIP_OFFLOAD_ARCH = amdgcnspirv

# -include hip/hip_runtime.h gives the kernels the built-in symbols (blockIdx,
# threadIdx, __global__, atomicAdd, float3, ...) that nvcc injects implicitly
# but hipcc requires to be included; the .cu sources stay CUDA-spelled and
# unmodified.
# -I. lets the CUDA-spelled #include <cuComplex.h> resolve to the local compat
# shim that maps cuComplex names to hipComplex.
# -Wno-bitwise-instead-of-logical: clang flags the intentional bitwise-or of
# booleans in the topological-charge stencils that nvcc accepts; keep -Werror
# for everything else.
HIPCCFLAGS = -O3 -Wall -Werror -Wno-bitwise-instead-of-logical --genco -include hip/hip_runtime.h -I.

WRAPPERS := $(CUDAFILES:.cu=_wrapper_hip.go)

%_wrapper_hip.go: %.cu cuda2go
	@ rm -f $(basename $<)*.co
	@ echo $(HIPCC) $(HIPCCFLAGS) --offload-arch=$(HIP_OFFLOAD_ARCH) $< -o $(basename $<).co
	@      $(HIPCC) $(HIPCCFLAGS) --offload-arch=$(HIP_OFFLOAD_ARCH) $< -o $(basename $<).co
	@ ./cuda2go -backend=hip $< > /dev/null
	@ gofmt -w -s -l $@ > /dev/null

else

# When CUDA_HOME is not an environment variable and is not set on the command line, use the nvcc compiler
# from the PATH
ifeq ($(CUDA_HOME),)
	NVCC=nvcc
else
	NVCC=${CUDA_HOME}/bin/nvcc
endif

# When CUDA_CC is not an environment variable and is not set on the command line, use compute capability 5.0
ifeq ($(CUDA_CC),)
	CUDA_CC = 50 # Lowest supported CC for mumax3.12
endif

# The gcc host compiler for nvcc
ifeq ($(NVCC_CCBIN),)
	override NVCC_CCBIN:=/usr/bin/gcc
endif

CUDA_VERSION := $(shell $(NVCC) --version | grep "Cuda compilation" | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | head -n 1)

NVCC_COMPATIBILITY_FLAGS := -std=c++03
ifneq (,$(filter 7.0 7.5 8.0,$(CUDA_VERSION)))
	NVCC_COMPATIBILITY_FLAGS :=
endif

NVCCFLAGS = $(NVCC_COMPATIBILITY_FLAGS) -ccbin=$(NVCC_CCBIN) --compiler-options -Werror --compiler-options -Wall -Xptxas -O3 -ptx

WRAPPERS := $(CUDAFILES:.cu=_wrapper.go)

%_wrapper.go: %.cu cuda2go
	@ rm -f $(basename $<)*.ptx
	@ for cc in $(CUDA_CC); do \
		echo $(NVCC) $(NVCCFLAGS) -arch=compute_$$cc -code=sm_$$cc $< -o $(basename $<)_$$cc.ptx ;\
		     $(NVCC) $(NVCCFLAGS) -arch=compute_$$cc -code=sm_$$cc $< -o $(basename $<)_$$cc.ptx ;\
	done
	@ ./cuda2go -backend=cuda $< > /dev/null
	@ gofmt -w -s -l $@ > /dev/null
	@ cp $(basename $<)_wrapper.go $(basename $<)_wrapper.go_linux_cuda${CUDA_VERSION}.tmp

endif


.PHONY: all wrappers clean realclean


all: wrappers
ifeq ($(BACKEND),hip)
	@echo "Built generic amdgcnspirv (SPIR-V) HIP images"
else
	@echo "Built with CUDA version ${CUDA_VERSION}"
endif
	go install -v


wrappers: $(WRAPPERS)


cuda2go: cuda2go.go
	go build $<


clean:
	rm -vf *.ptx *.co *.tmp


realclean:
	rm -vf *_wrapper.go *_wrapper_hip.go *.ptx *.co cuda2go
