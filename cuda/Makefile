# Makefile to build mumax3 cuda kernels and create GO wrappers
#
# The build process depends on two environment variables.
#
#
#      CUDA_HOME   The path to the cuda home directory. Can be used to compile against a specific cuda version.
#                  Fallback: "usr/local/cuda"
#
#
#      CUDA_CC     The list of compute capabilities for which the kernel will be build.
#                  Fallback: "30"
#
#                  Different CUDA versions support different compute capabilities, as shown in the list below.
#                  CUDA SDK  7.5 support for compute capability 20 21 30 32 35 37 50 52 53
#                  CUDA SDK  8.0 support for compute capability 20 21 30 32 35 37 50 52 53 60 61 62
#                  CUDA SDK  9.0 support for compute capability       30 32 35 37 50 52 53 60 61 62 70
#                  CUDA SDK  9.1 support for compute capability       30 32 35 37 50 52 53 60 61 62 70
#                  CUDA SDK  9.2 support for compute capability       30 32 35 37 50 52 53 60 61 62 70 72
#                  CUDA SDK 10.0 support for compute capability       30 32 35 37 50 52 53 60 61 62 70 72 75
#
#
#
# Example: build a "fat" wrapper against cuda 9.0
#
# 	make CUDA_HOME=/usr/local/cuda-9.0 CUDA_CC="30 32 35 37 50 52 53 60 61 62 70"
#


SHELL = /bin/bash

# When CUDA_HOME is not an environment variable and is not set on the command line, use "/usr/local/cuda"
ifeq ($(CUDA_HOME),)
    CUDA_HOME = /usr/local/cuda
endif

# When CUDA_CC is not an environment variable and is not set on the command line, use compute capability 3.0
ifeq ($(CUDA_CC),)
    CUDA_CC = 30
endif

NVCC = $(CUDA_HOME)/bin/nvcc
NVCCFLAGS = -I$(CUDA_HOME)/include --compiler-options -Werror --compiler-options -Wall -Xptxas -O3 -ptx

CUDAFILES := $(wildcard *.cu)
WRAPPERS := $(CUDAFILES:.cu=_wrapper.go)



.PHONY: all wrappers cuda2g0 clean realclean

all: wrappers
	go install -v

wrappers: $(WRAPPERS)

%_wrapper.go: %.cu cuda2go
	@ rm -f $(basename $<)*.ptx
	@ for cc in $(CUDA_CC); do \
		echo $(NVCC) $(NVCCFLAGS) -arch=compute_$$cc -code=sm_$$cc $< -o $(basename $<)_$$cc.ptx ;\
		     $(NVCC) $(NVCCFLAGS) -arch=compute_$$cc -code=sm_$$cc $< -o $(basename $<)_$$cc.ptx ;\
	done
	@ ./cuda2go $< > /dev/null
	@ gofmt -w -s -l $@ > /dev/null

cuda2go:
	go build cuda2go.go

clean:
	rm -f *.ptx

realclean:
	rm -f *_wrapper.go *.ptx cuda2go
