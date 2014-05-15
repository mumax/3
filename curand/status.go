package curand

//#include <curand.h>
import "C"

import (
//"unsafe"
)

type Status int

const (
	SUCCESS               Status = C.CURAND_STATUS_SUCCESS               // No errors
	VERSION_MISMATCH      Status = C.CURAND_STATUS_VERSION_MISMATCH      // Header file and linked library version do not match
	NOT_INITIALIZED       Status = C.CURAND_STATUS_NOT_INITIALIZED       // Generator not initialized
	ALLOCATION_FAILED     Status = C.CURAND_STATUS_ALLOCATION_FAILED     // Memory allocation failed
	TYPE_ERROR            Status = C.CURAND_STATUS_TYPE_ERROR            // Generator is wrong type
	OUT_OF_RANGE          Status = C.CURAND_STATUS_OUT_OF_RANGE          // Argument out of range
	LENGTH_NOT_MULTIPLE   Status = C.CURAND_STATUS_LENGTH_NOT_MULTIPLE   // Length requested is not a multple of dimension
	LAUNCH_FAILURE        Status = C.CURAND_STATUS_LAUNCH_FAILURE        // Kernel launch failure
	PREEXISTING_FAILURE   Status = C.CURAND_STATUS_PREEXISTING_FAILURE   // Preexisting failure on library entry
	INITIALIZATION_FAILED Status = C.CURAND_STATUS_INITIALIZATION_FAILED // Initialization of CUDA failed
	ARCH_MISMATCH         Status = C.CURAND_STATUS_ARCH_MISMATCH         // Architecture mismatch, GPU does not support requested feature
	INTERNAL_ERROR        Status = C.CURAND_STATUS_INTERNAL_ERROR        // Internal library error
)

// Documentation was taken from the curand headers.
