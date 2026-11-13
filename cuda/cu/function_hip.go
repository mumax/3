//go:build hip

package cu

// This file implements manipulations on HIP functions

//#include <hip/hip_runtime.h>
import "C"

import (
	"unsafe"
)

// Represents a hipFunction_t, a reference to a function within a module.
type Function uintptr

func FuncGetAttribute(attrib FunctionAttribute, function Function) int {
	var attr C.int
	err := Result(C.hipFuncGetAttribute(&attr, C.hipFunction_attribute(attrib), C.hipFunction_t(unsafe.Pointer(uintptr(function)))))
	if err != SUCCESS {
		panic(err)
	}
	return int(attr)
}

func (f Function) GetAttribute(attrib FunctionAttribute) int {
	return FuncGetAttribute(attrib, f)
}

type FunctionAttribute int

const (
	FUNC_A_MAX_THREADS_PER_BLOCK FunctionAttribute = C.HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK // The maximum number of threads per block, beyond which a launch of the function would fail.
	FUNC_A_SHARED_SIZE_BYTES     FunctionAttribute = C.HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES     // The size in bytes of statically-allocated shared memory required by this function.
	FUNC_A_CONST_SIZE_BYTES      FunctionAttribute = C.HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES      // The size in bytes of user-allocated constant memory required by this function.
	FUNC_A_LOCAL_SIZE_BYTES      FunctionAttribute = C.HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES      // The size in bytes of local memory used by each thread of this function.
	FUNC_A_NUM_REGS              FunctionAttribute = C.HIP_FUNC_ATTRIBUTE_NUM_REGS              // The number of registers used by each thread of this function.
	FUNC_A_PTX_VERSION           FunctionAttribute = C.HIP_FUNC_ATTRIBUTE_PTX_VERSION           // The PTX virtual architecture version for which the function was compiled.
	FUNC_A_BINARY_VERSION        FunctionAttribute = C.HIP_FUNC_ATTRIBUTE_BINARY_VERSION        // The binary architecture version for which the function was compiled.
)
