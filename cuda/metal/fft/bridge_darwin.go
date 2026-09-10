//go:build darwin && arm64 && cgo
// +build darwin,arm64,cgo

package fft

/*
#cgo darwin,arm64 CFLAGS: -mmacosx-version-min=14.0
#cgo darwin,arm64 CXXFLAGS: -x objective-c++ -std=c++17 -fobjc-arc -fblocks -mmacosx-version-min=14.0
#cgo darwin,arm64 LDFLAGS: -mmacosx-version-min=14.0 -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph
#include "bridge.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/mumax/3/cuda/metal"
)

// Transform uses cuFFT's public numeric values so the compatibility package
// can pass its Type through without a translation table.
type Transform int32

const (
	ComplexToComplex Transform = C.MF_C2C
	RealToComplex    Transform = C.MF_R2C
	ComplexToReal    Transform = C.MF_C2R
)

// CreatePlan creates a cached MPSGraph FFT plan.
func CreatePlan(layout Layout, transform Transform) (uintptr, error) {
	if err := metal.Initialize(); err != nil {
		return 0, fmt.Errorf("metal fft: initialize runtime: %w", err)
	}
	dimensions := make([]C.int64_t, len(layout.Dimensions))
	for i, dimension := range layout.Dimensions {
		dimensions[i] = C.int64_t(dimension)
	}
	var message *C.char
	handle := C.mf_plan_create(
		&dimensions[0],
		C.size_t(len(dimensions)),
		C.int64_t(layout.Batch),
		C.int32_t(transform),
		&message,
	)
	runtime.KeepAlive(dimensions)
	if handle == nil {
		return 0, bridgeError("create plan", message)
	}
	if message != nil {
		C.mf_free_error(message)
	}
	return uintptr(handle), nil
}

// Execute appends a transform to the Metal runtime's current command buffer.
// It deliberately does not commit: Flush/Sync remain the sole ownership point
// for CUDA-stream-compatible batching.
func Execute(handle, input, output uintptr, direction int) error {
	if handle == 0 || input == 0 || output == 0 {
		return fmt.Errorf("metal fft: invalid nil plan or buffer")
	}
	var message *C.char
	status := C.mf_plan_execute(
		unsafe.Pointer(handle),
		unsafe.Pointer(input),
		unsafe.Pointer(output),
		C.int32_t(direction),
		&message,
	)
	if status != C.MF_SUCCESS {
		return bridgeError("execute", message)
	}
	if message != nil {
		C.mf_free_error(message)
	}
	return nil
}

// DestroyPlan releases graph and descriptor objects cached by a plan.
func DestroyPlan(handle uintptr) error {
	if handle == 0 {
		return nil
	}
	// MPSGraph command buffers may retain references into the graph until GPU
	// completion. cuFFT plan destruction is rare, so synchronize here before
	// releasing the retained graph rather than risking an asynchronous
	// use-after-free.
	syncErr := metal.Sync()
	var message *C.char
	status := C.mf_plan_destroy(unsafe.Pointer(handle), &message)
	if status != C.MF_SUCCESS {
		return bridgeError("destroy plan", message)
	}
	if message != nil {
		C.mf_free_error(message)
	}
	if syncErr != nil {
		return fmt.Errorf("metal fft: synchronize before destroying plan: %w", syncErr)
	}
	return nil
}

func bridgeError(operation string, message *C.char) error {
	text := "unknown failure"
	if message != nil {
		text = C.GoString(message)
		C.mf_free_error(message)
	}
	return fmt.Errorf("metal fft: %s: %s", operation, text)
}
