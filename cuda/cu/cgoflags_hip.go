//go:build hip

package cu

// This file provides CGO flags to find the HIP runtime/driver library and headers.

//#cgo CFLAGS: -I/opt/rocm/include -D__HIP_PLATFORM_AMD__
//#cgo LDFLAGS: -L/opt/rocm/lib -lamdhip64
import "C"
