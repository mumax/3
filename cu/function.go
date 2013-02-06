package cu

// This file implements manipulations on CUDA functions

//#include <cuda.h>
import "C"

import (
	"unsafe"
)

// Represents a CUDA CUfunction, a reference to a function within a module.
type Function uintptr

func FuncGetAttribute(attrib FunctionAttribute, function Function) int {
	var attr C.int
	err := Result(C.cuFuncGetAttribute(&attr, C.CUfunction_attribute(attrib), C.CUfunction(unsafe.Pointer(uintptr(function)))))
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
	FUNC_A_MAX_THREADS_PER_BLOCK FunctionAttribute = C.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK // The maximum number of threads per block, beyond which a launch of the function would fail.
	FUNC_A_SHARED_SIZE_BYTES     FunctionAttribute = C.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES     // The size in bytes of statically-allocated shared memory required by this function.
	FUNC_A_CONST_SIZE_BYTES      FunctionAttribute = C.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES      // The size in bytes of user-allocated constant memory required by this function.
	FUNC_A_LOCAL_SIZE_BYTES      FunctionAttribute = C.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES      // The size in bytes of local memory used by each thread of this function.
	FUNC_A_NUM_REGS              FunctionAttribute = C.CU_FUNC_ATTRIBUTE_NUM_REGS              // The number of registers used by each thread of this function.
	FUNC_A_PTX_VERSION           FunctionAttribute = C.CU_FUNC_ATTRIBUTE_PTX_VERSION           // The PTX virtual architecture version for which the function was compiled.
	FUNC_A_BINARY_VERSION        FunctionAttribute = C.CU_FUNC_ATTRIBUTE_BINARY_VERSION        // The binary architecture version for which the function was compiled.
)
