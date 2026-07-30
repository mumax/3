
# Use the default go compiler
GO_BUILDFLAGS=-compiler gc
# Or uncomment the line below to use the gccgo compiler, which may 
# or may not be faster than gc and which may or may not compile...
# GO_BUILDFLAGS=-compiler gccgo -gccgoflags '-static-libgcc -O4 -Ofast -march=native'

CGO_CFLAGS_ALLOW='(-fno-schedule-insns|-malign-double|-ffast-math)'

HOST_OS := $(shell go env GOOS)
HOST_ARCH := $(shell go env GOARCH)
ifeq ($(HOST_OS)-$(HOST_ARCH),darwin-arm64)
	GPU_KERNELS := metalkernels
else
	GPU_KERNELS := cudakernels
endif

.PHONY: all cudakernels metalkernels check-metal metal-shaders clean realclean checktests runtests hooks


all: $(GPU_KERNELS) hooks
	go install -v $(GO_BUILDFLAGS) github.com/mumax/3/...
	cd cmd/mumax3/ && $(MAKE)

cudakernels:
	cd cuda && $(MAKE) NVCC_CCBIN=$(NVCC_CCBIN)

metalkernels:
	@test -s cuda/metal/kernels/mumax3_kernels.metal || \
		(echo "Generated Metal kernels are missing; run make check-metal" >&2; exit 1)

# Compile the generated MSL with the active macOS SDK when the offline Metal
# compiler is installed. MuMax3 also compiles this source through Metal at
# runtime, so Xcode Command Line Tools alone remain sufficient for end users.
metal-shaders: metalkernels
	@set -eu; \
	if xcrun -f metal >/dev/null 2>&1 && xcrun -f metallib >/dev/null 2>&1; then \
		metal_tmp=$$(mktemp -d "$${TMPDIR:-/tmp}/mumax3-metal.XXXXXX"); \
		trap 'rm -r "$$metal_tmp"' EXIT; \
		xcrun -sdk macosx metal -std=metal3.0 -c cuda/metal/kernels/mumax3_kernels.metal -o "$$metal_tmp/mumax3_kernels.air"; \
		xcrun -sdk macosx metallib "$$metal_tmp/mumax3_kernels.air" -o "$$metal_tmp/mumax3_kernels.metallib"; \
		echo "Metal shader library compiled successfully"; \
	else \
		echo "Offline Metal compiler not installed; runtime compilation will be used"; \
	fi

check-metal: metal-shaders
	python3 cuda/metal/cmd/cuda2metal/generate.py --check
	python3 -m unittest discover -s cuda/metal/cmd/cuda2metal -p 'test_*.py'
	go test ./cuda/metal/...
	go test ./cuda ./cuda/cu ./cuda/cufft ./cuda/curand
	go build ./...

doc:
	cd doc && $(MAKE)

test: all
	go test -vet=off -i github.com/mumax/3/...
	go test -vet=off $(PKGS)  github.com/mumax/3/...
	cd test && ./run.bash

hooks: .git/hooks/post-commit .git/hooks/pre-commit

.git/hooks/post-commit: post-commit
	ln -sf $(CURDIR)/$< $@

.git/hooks/pre-commit: pre-commit
	ln -sf $(CURDIR)/$< $@

clean:
	rm -frv $(GOPATH)/pkg/*/github.com/mumax/3/*
	rm -frv $(GOPATH)/bin/mumax3*
	cd cuda && $(MAKE) clean

realclean: clean
	cd cuda && ${MAKE} realclean
