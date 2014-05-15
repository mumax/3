package cufft

//#include <cufft.h>
import "C"

import (
	"fmt"
)

// FFT result
type Result int

// FFT result value
const (
	SUCCESS                   Result = C.CUFFT_SUCCESS
	INVALID_PLAN              Result = C.CUFFT_INVALID_PLAN
	ALLOC_FAILED              Result = C.CUFFT_ALLOC_FAILED
	INVALID_TYPE              Result = C.CUFFT_INVALID_TYPE
	INVALID_VALUE             Result = C.CUFFT_INVALID_VALUE
	INTERNAL_ERROR            Result = C.CUFFT_INTERNAL_ERROR
	EXEC_FAILED               Result = C.CUFFT_EXEC_FAILED
	SETUP_FAILED              Result = C.CUFFT_SETUP_FAILED
	INVALID_SIZE              Result = C.CUFFT_INVALID_SIZE
	UNALIGNED_DATA            Result = C.CUFFT_UNALIGNED_DATA
	INCOMPLETE_PARAMETER_LIST Result = 0xA // cuda6 values copied to avoid dependency on cuda6/cufft.h
	INVALID_DEVICE            Result = 0xB
	PARSE_ERROR               Result = 0xC
	NO_WORKSPACE              Result = 0xD
)

func (r Result) String() string {
	if str, ok := resultString[r]; ok {
		return str
	}
	return fmt.Sprint("CUFFT Result with unknown error number:", int(r))
}

var resultString map[Result]string = map[Result]string{
	SUCCESS:                   "CUFFT_SUCCESS",
	INVALID_PLAN:              "CUFFT_INVALID_PLAN",
	ALLOC_FAILED:              "CUFFT_ALLOC_FAILED",
	INVALID_TYPE:              "CUFFT_INVALID_TYPE",
	INVALID_VALUE:             "CUFFT_INVALID_VALUE",
	INTERNAL_ERROR:            "CUFFT_INTERNAL_ERROR",
	EXEC_FAILED:               "CUFFT_EXEC_FAILED",
	SETUP_FAILED:              "CUFFT_SETUP_FAILED",
	INVALID_SIZE:              "CUFFT_INVALID_SIZE",
	UNALIGNED_DATA:            "CUFFT_UNALIGNED_DATA",
	INCOMPLETE_PARAMETER_LIST: "CUFFT_INCOMPLETE_PARAMETER_LIST",
	INVALID_DEVICE:            "CUFFT_INVALID_DEVICE",
	PARSE_ERROR:               "CUFFT_PARSE_ERROR",
	NO_WORKSPACE:              "CUFFT_NO_WORKSPACE"}
