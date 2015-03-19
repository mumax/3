package curand

//#include <curand.h>
import "C"

import (
	"fmt"
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

func (s Status) String() string {
	if str, ok := statusStr[s]; ok {
		return str
	} else {
		return fmt.Sprint("CURAND ERROR NUMBER ", int(s))
	}
}

var statusStr = map[Status]string{
	SUCCESS:               "CURAND_STATUS_SUCCESS",
	VERSION_MISMATCH:      "CURAND_STATUS_VERSION_MISMATCH",
	NOT_INITIALIZED:       "CURAND_STATUS_NOT_INITIALIZED",
	ALLOCATION_FAILED:     "CURAND_STATUS_ALLOCATION_FAILED",
	TYPE_ERROR:            "CURAND_STATUS_TYPE_ERROR",
	OUT_OF_RANGE:          "CURAND_STATUS_OUT_OF_RANGE",
	LENGTH_NOT_MULTIPLE:   "CURAND_STATUS_LENGTH_NOT_MULTIPLE",
	LAUNCH_FAILURE:        "CURAND_STATUS_LAUNCH_FAILURE",
	PREEXISTING_FAILURE:   "CURAND_STATUS_PREEXISTING_FAILURE",
	INITIALIZATION_FAILED: "CURAND_STATUS_INITIALIZATION_FAILED",
	ARCH_MISMATCH:         "CURAND_STATUS_ARCH_MISMATCH",
	INTERNAL_ERROR:        "CURAND_STATUS_INTERNAL_ERROR",
}

// Documentation was taken from the curand headers.
