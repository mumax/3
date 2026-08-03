//go:build hip

package curand

// See generator.go for the shim rationale.

//#include "curand_shim.h"
import "C"

import (
	"fmt"
)

type Status int

const (
	SUCCESS               Status = C.HIPRAND_STATUS_SUCCESS               // No errors
	VERSION_MISMATCH      Status = C.HIPRAND_STATUS_VERSION_MISMATCH      // Header file and linked library version do not match
	NOT_INITIALIZED       Status = C.HIPRAND_STATUS_NOT_INITIALIZED       // Generator not initialized
	ALLOCATION_FAILED     Status = C.HIPRAND_STATUS_ALLOCATION_FAILED     // Memory allocation failed
	TYPE_ERROR            Status = C.HIPRAND_STATUS_TYPE_ERROR            // Generator is wrong type
	OUT_OF_RANGE          Status = C.HIPRAND_STATUS_OUT_OF_RANGE          // Argument out of range
	LENGTH_NOT_MULTIPLE   Status = C.HIPRAND_STATUS_LENGTH_NOT_MULTIPLE   // Length requested is not a multiple of dimension
	LAUNCH_FAILURE        Status = C.HIPRAND_STATUS_LAUNCH_FAILURE        // Kernel launch failure
	PREEXISTING_FAILURE   Status = C.HIPRAND_STATUS_PREEXISTING_FAILURE   // Preexisting failure on library entry
	INITIALIZATION_FAILED Status = C.HIPRAND_STATUS_INITIALIZATION_FAILED // Initialization of HIP failed
	ARCH_MISMATCH         Status = C.HIPRAND_STATUS_ARCH_MISMATCH         // Architecture mismatch, GPU does not support requested feature
	INTERNAL_ERROR        Status = C.HIPRAND_STATUS_INTERNAL_ERROR        // Internal library error
)

func (s Status) String() string {
	if str, ok := statusStr[s]; ok {
		return str
	}
	return fmt.Sprint("HIPRAND ERROR NUMBER ", int(s))
}

var statusStr = map[Status]string{
	SUCCESS:               "HIPRAND_STATUS_SUCCESS",
	VERSION_MISMATCH:      "HIPRAND_STATUS_VERSION_MISMATCH",
	NOT_INITIALIZED:       "HIPRAND_STATUS_NOT_INITIALIZED",
	ALLOCATION_FAILED:     "HIPRAND_STATUS_ALLOCATION_FAILED",
	TYPE_ERROR:            "HIPRAND_STATUS_TYPE_ERROR",
	OUT_OF_RANGE:          "HIPRAND_STATUS_OUT_OF_RANGE",
	LENGTH_NOT_MULTIPLE:   "HIPRAND_STATUS_LENGTH_NOT_MULTIPLE",
	LAUNCH_FAILURE:        "HIPRAND_STATUS_LAUNCH_FAILURE",
	PREEXISTING_FAILURE:   "HIPRAND_STATUS_PREEXISTING_FAILURE",
	INITIALIZATION_FAILED: "HIPRAND_STATUS_INITIALIZATION_FAILED",
	ARCH_MISMATCH:         "HIPRAND_STATUS_ARCH_MISMATCH",
	INTERNAL_ERROR:        "HIPRAND_STATUS_INTERNAL_ERROR",
}

// Documentation was taken from the hipRAND headers.
