//go:build darwin && arm64 && cgo

package metal

/*
#cgo darwin,arm64 CFLAGS: -mmacosx-version-min=14.0
#cgo darwin,arm64 CXXFLAGS: -x objective-c++ -std=c++17 -fobjc-arc -fblocks -mmacosx-version-min=14.0
#cgo darwin,arm64 LDFLAGS: -mmacosx-version-min=14.0 -framework Foundation -framework Metal
#include <stdlib.h>
#include "metal_runtime.h"
*/
import "C"

import (
	"errors"
	"unsafe"
)

func Available() bool {
	return Initialize() == nil
}

func Initialize() error {
	return runtimeStatus(C.mr_initialize(nil), nil)
}

func Close() error {
	var message *C.char
	return runtimeStatus(C.mr_shutdown(&message), message)
}

func Info() (DeviceInfo, error) {
	var info C.mr_device_info
	var message *C.char
	status := C.mr_get_device_info(&info, &message)
	if err := runtimeStatus(status, message); err != nil {
		return DeviceInfo{}, err
	}
	return DeviceInfo{
		Name:                         C.GoString(&info.name[0]),
		UnifiedMemory:                info.unified_memory != 0,
		MaxThreadsPerThreadgroup:     [3]uint64{uint64(info.max_threads_x), uint64(info.max_threads_y), uint64(info.max_threads_z)},
		RecommendedMaxWorkingSetSize: uint64(info.recommended_max_working_set_size),
		CurrentAllocatedSize:         uint64(info.current_allocated_size),
		TrackedAllocationSize:        uint64(info.tracked_allocation_size),
		TrackedPeakAllocationSize:    uint64(info.tracked_peak_allocation_size),
	}, nil
}

func RegisterSource(source string) error {
	if source == "" {
		return errors.New("mumax3/metal: refusing to register empty Metal source")
	}
	data := C.CBytes([]byte(source))
	defer C.free(data)
	var message *C.char
	return runtimeStatus(
		C.mr_register_source(data, C.size_t(len(source)), &message),
		message,
	)
}

func RegisterLibrary(library []byte) error {
	if len(library) == 0 {
		return errors.New("mumax3/metal: refusing to register an empty metallib")
	}
	data := C.CBytes(library)
	defer C.free(data)
	var message *C.char
	return runtimeStatus(
		C.mr_register_library(data, C.size_t(len(library)), &message),
		message,
	)
}

func Alloc(bytes int64) (unsafe.Pointer, error) {
	if bytes <= 0 {
		return nil, errors.New("mumax3/metal: allocation size must be positive")
	}
	if uint64(bytes) > uint64(^C.size_t(0)) {
		return nil, errors.New("mumax3/metal: allocation size exceeds platform size_t")
	}
	var pointer unsafe.Pointer
	var message *C.char
	status := C.mr_alloc(C.size_t(bytes), &pointer, &message)
	return pointer, runtimeStatus(status, message)
}

func MustAlloc(bytes int64) unsafe.Pointer {
	pointer, err := Alloc(bytes)
	if err != nil {
		panic(err)
	}
	return pointer
}

func Free(pointer unsafe.Pointer) error {
	if pointer == nil {
		return nil
	}
	var message *C.char
	return runtimeStatus(C.mr_free(pointer, &message), message)
}

func AllocationRange(pointer unsafe.Pointer) (unsafe.Pointer, int64, error) {
	if pointer == nil {
		return nil, 0, errors.New("mumax3/metal: allocation query pointer is nil")
	}
	var base unsafe.Pointer
	var bytes C.size_t
	var message *C.char
	status := C.mr_get_address_range(pointer, &base, &bytes, &message)
	if err := runtimeStatus(status, message); err != nil {
		return nil, 0, err
	}
	return base, int64(bytes), nil
}

func Copy(dst, src unsafe.Pointer, bytes int64) error {
	if bytes == 0 {
		return nil
	}
	if err := validCopy(dst, src, bytes); err != nil {
		return err
	}
	var message *C.char
	return runtimeStatus(C.mr_copy(dst, src, C.size_t(bytes), &message), message)
}

func CopyToDevice(dst, src unsafe.Pointer, bytes int64) error {
	if bytes == 0 {
		return nil
	}
	if err := validCopy(dst, src, bytes); err != nil {
		return err
	}
	var message *C.char
	return runtimeStatus(C.mr_copy_to_device(dst, src, C.size_t(bytes), &message), message)
}

func CopyToHost(dst, src unsafe.Pointer, bytes int64) error {
	if bytes == 0 {
		return nil
	}
	if err := validCopy(dst, src, bytes); err != nil {
		return err
	}
	var message *C.char
	return runtimeStatus(C.mr_copy_to_host(dst, src, C.size_t(bytes), &message), message)
}

func Fill(dst unsafe.Pointer, value byte, bytes int64) error {
	if bytes == 0 {
		return nil
	}
	if dst == nil || bytes < 0 {
		return errors.New("mumax3/metal: invalid fill range")
	}
	var message *C.char
	return runtimeStatus(C.mr_fill(dst, C.uint8_t(value), C.size_t(bytes), &message), message)
}

func FillUint32(dst unsafe.Pointer, value uint32, count int64) error {
	if count == 0 {
		return nil
	}
	if dst == nil || count < 0 {
		return errors.New("mumax3/metal: invalid uint32 fill range")
	}
	var message *C.char
	return runtimeStatus(
		C.mr_fill_u32(dst, C.uint32_t(value), C.size_t(count), &message),
		message,
	)
}

func Launch(name string, cfg GridConfig, args ...Arg) error {
	if name == "" {
		return errors.New("mumax3/metal: kernel name is empty")
	}
	if cfg.BlockX == 0 || cfg.BlockY == 0 || cfg.BlockZ == 0 {
		return errors.New("mumax3/metal: threadgroup dimensions must be non-zero")
	}
	if len(args) > 31 {
		return errors.New("mumax3/metal: a Metal compute kernel cannot bind more than 31 buffer-table arguments")
	}

	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	cargs := make([]C.mr_arg, len(args))
	for i, arg := range args {
		cargs[i].kind = C.uint32_t(arg.kind)
		cargs[i].size = C.uint32_t(arg.size)
		cargs[i].buffer = arg.pointer
		cargs[i].bits = C.uint64_t(arg.bits)
	}

	grid := C.mr_grid{
		grid_x:  C.uint32_t(cfg.GridX),
		grid_y:  C.uint32_t(cfg.GridY),
		grid_z:  C.uint32_t(cfg.GridZ),
		block_x: C.uint32_t(cfg.BlockX),
		block_y: C.uint32_t(cfg.BlockY),
		block_z: C.uint32_t(cfg.BlockZ),
	}

	var cargsPointer *C.mr_arg
	if len(cargs) != 0 {
		cargsPointer = &cargs[0]
	}
	var message *C.char
	return runtimeStatus(
		C.mr_launch(cname, grid, cargsPointer, C.size_t(len(cargs)), &message),
		message,
	)
}

func MustLaunch(name string, cfg GridConfig, args ...Arg) {
	if err := Launch(name, cfg, args...); err != nil {
		panic(err)
	}
}

func Flush() error {
	var message *C.char
	return runtimeStatus(C.mr_flush(&message), message)
}

func Sync() error {
	var message *C.char
	return runtimeStatus(C.mr_synchronize(&message), message)
}

func validCopy(dst, src unsafe.Pointer, bytes int64) error {
	if dst == nil || src == nil || bytes < 0 {
		return errors.New("mumax3/metal: invalid copy range")
	}
	return nil
}

func runtimeStatus(status C.int, message *C.char) error {
	if message != nil {
		defer C.mr_free_error(message)
	}
	if status == C.MR_SUCCESS {
		return nil
	}
	text := "unknown Metal runtime failure"
	if message != nil {
		text = C.GoString(message)
	}
	return &RuntimeError{Code: int(status), Message: text}
}
