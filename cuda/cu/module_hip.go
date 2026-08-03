//go:build hip

package cu

// This file implements loading of HIP code object modules

//#include <hip/hip_runtime.h>
//#include <stdlib.h>
import "C"

import (
	"unsafe"
)

// Represents a hipModule_t, a reference to executable device code.
type Module uintptr

// Loads a compute module from file
func ModuleLoad(fname string) Module {
	var mod C.hipModule_t
	cstr := C.CString(fname)
	defer C.free(unsafe.Pointer(cstr))
	err := Result(C.hipModuleLoad(&mod, cstr))
	if err != SUCCESS {
		panic(err)
	}
	return Module(uintptr(unsafe.Pointer(mod)))
}

// Loads a compute module from an in-memory code object image.
// The image is a binary HIP code object (with NUL bytes), so it is passed
// as a length-delimited byte slice rather than a NUL-terminated C string.
func ModuleLoadData(image []byte) Module {
	var mod C.hipModule_t
	err := Result(C.hipModuleLoadData(&mod, unsafe.Pointer(&image[0])))
	if err != SUCCESS {
		panic(err)
	}
	return Module(uintptr(unsafe.Pointer(mod)))
}

// Returns a Function handle.
func ModuleGetFunction(module Module, name string) Function {
	var function C.hipFunction_t
	cstr := C.CString(name)
	defer C.free(unsafe.Pointer(cstr))
	err := Result(C.hipModuleGetFunction(
		&function,
		C.hipModule_t(unsafe.Pointer(uintptr(module))),
		cstr))
	if err != SUCCESS {
		panic(err)
	}
	return Function(uintptr(unsafe.Pointer(function)))
}

// Returns a Function handle.
func (m Module) GetFunction(name string) Function {
	return ModuleGetFunction(m, name)
}
