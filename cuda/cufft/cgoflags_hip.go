//go:build hip

package cufft

// This file provides CGO flags to find the hipFFT library and headers.

//#cgo CFLAGS: -I/opt/rocm/include -D__HIP_PLATFORM_AMD__
//#cgo LDFLAGS: -L/opt/rocm/lib -lhipfft
import "C"
