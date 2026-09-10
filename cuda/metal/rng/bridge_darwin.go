//go:build darwin && arm64 && cgo
// +build darwin,arm64,cgo

package rng

/*
#cgo darwin,arm64 CFLAGS: -mmacosx-version-min=14.0
#cgo darwin,arm64 CXXFLAGS: -x objective-c++ -std=c++17 -fobjc-arc -fblocks -mmacosx-version-min=14.0
#cgo darwin,arm64 LDFLAGS: -mmacosx-version-min=14.0 -framework Foundation -framework Metal
#include "bridge.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	_ "github.com/mumax/3/cuda/metal"
)

// CreateGenerator constructs a Philox-backed Metal generator.
func CreateGenerator(rngType int) (uintptr, error) {
	var message *C.char
	generator := C.mrg_generator_create(C.int32_t(rngType), &message)
	if generator == nil {
		return 0, nativeError("create generator", message)
	}
	if message != nil {
		C.mrg_free_error(message)
	}
	return uintptr(generator), nil
}

// SetSeed resets both the Philox key and its monotonically increasing counter.
func SetSeed(generator uintptr, seed uint64) error {
	if generator == 0 {
		return fmt.Errorf("metal rng: nil generator")
	}
	var message *C.char
	status := C.mrg_generator_set_seed(unsafe.Pointer(generator), C.uint64_t(seed), &message)
	if status != C.MRG_SUCCESS {
		return nativeError("set seed", message)
	}
	if message != nil {
		C.mrg_free_error(message)
	}
	return nil
}

// GenerateNormal appends a Philox plus Box-Muller kernel to the current Metal
// command buffer. The generator reserves one 128-bit counter per four outputs.
func GenerateNormal(generator, output uintptr, count int64, mean, standardDeviation float32) error {
	if generator == 0 || output == 0 {
		return fmt.Errorf("metal rng: nil generator or output")
	}
	var message *C.char
	status := C.mrg_generate_normal(
		unsafe.Pointer(generator),
		unsafe.Pointer(output),
		C.int64_t(count),
		C.float(mean),
		C.float(standardDeviation),
		&message,
	)
	if status != C.MRG_SUCCESS {
		return nativeError("generate normal", message)
	}
	if message != nil {
		C.mrg_free_error(message)
	}
	return nil
}

// GenerateRaw appends raw Philox uint4 blocks. It is primarily a diagnostic
// API for cross-device known-answer tests; normal generation uses the same
// integer core and advances the same counter.
func GenerateRaw(generator, output uintptr, blockCount int64) error {
	if generator == 0 || output == 0 {
		return fmt.Errorf("metal rng: nil generator or output")
	}
	var message *C.char
	status := C.mrg_generate_raw(
		unsafe.Pointer(generator),
		unsafe.Pointer(output),
		C.int64_t(blockCount),
		&message,
	)
	if status != C.MRG_SUCCESS {
		return nativeError("generate raw Philox blocks", message)
	}
	if message != nil {
		C.mrg_free_error(message)
	}
	return nil
}

func nativeError(operation string, message *C.char) error {
	text := "unknown failure"
	if message != nil {
		text = C.GoString(message)
		C.mrg_free_error(message)
	}
	return fmt.Errorf("metal rng: %s: %s", operation, text)
}
