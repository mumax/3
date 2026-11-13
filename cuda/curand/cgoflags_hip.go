//go:build hip

package curand

// This file provides CGO flags to find the hipRAND library and headers.

//#cgo CFLAGS: -I/opt/rocm/include -D__HIP_PLATFORM_AMD__
//#cgo LDFLAGS: -L/opt/rocm/lib -lhiprand
import "C"
