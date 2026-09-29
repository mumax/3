//go:build hip

package cufft

//#include <hipfft/hipfft.h>
import "C"

import (
	"fmt"
)

// FFT result
type Result int

// FFT result value
const (
	SUCCESS                   Result = C.HIPFFT_SUCCESS
	INVALID_PLAN              Result = C.HIPFFT_INVALID_PLAN
	ALLOC_FAILED              Result = C.HIPFFT_ALLOC_FAILED
	INVALID_TYPE              Result = C.HIPFFT_INVALID_TYPE
	INVALID_VALUE             Result = C.HIPFFT_INVALID_VALUE
	INTERNAL_ERROR            Result = C.HIPFFT_INTERNAL_ERROR
	EXEC_FAILED               Result = C.HIPFFT_EXEC_FAILED
	SETUP_FAILED              Result = C.HIPFFT_SETUP_FAILED
	INVALID_SIZE              Result = C.HIPFFT_INVALID_SIZE
	UNALIGNED_DATA            Result = C.HIPFFT_UNALIGNED_DATA
	INCOMPLETE_PARAMETER_LIST Result = C.HIPFFT_INCOMPLETE_PARAMETER_LIST
	INVALID_DEVICE            Result = C.HIPFFT_INVALID_DEVICE
	PARSE_ERROR               Result = C.HIPFFT_PARSE_ERROR
	NO_WORKSPACE              Result = C.HIPFFT_NO_WORKSPACE
)

func (r Result) String() string {
	if str, ok := resultString[r]; ok {
		return str
	}
	return fmt.Sprint("FFT Result with unknown error number:", int(r))
}

var resultString = map[Result]string{
	SUCCESS:                   "HIPFFT_SUCCESS",
	INVALID_PLAN:              "HIPFFT_INVALID_PLAN",
	ALLOC_FAILED:              "HIPFFT_ALLOC_FAILED",
	INVALID_TYPE:              "HIPFFT_INVALID_TYPE",
	INVALID_VALUE:             "HIPFFT_INVALID_VALUE",
	INTERNAL_ERROR:            "HIPFFT_INTERNAL_ERROR",
	EXEC_FAILED:               "HIPFFT_EXEC_FAILED",
	SETUP_FAILED:              "HIPFFT_SETUP_FAILED",
	INVALID_SIZE:              "HIPFFT_INVALID_SIZE",
	UNALIGNED_DATA:            "HIPFFT_UNALIGNED_DATA",
	INCOMPLETE_PARAMETER_LIST: "HIPFFT_INCOMPLETE_PARAMETER_LIST",
	INVALID_DEVICE:            "HIPFFT_INVALID_DEVICE",
	PARSE_ERROR:               "HIPFFT_PARSE_ERROR",
	NO_WORKSPACE:              "HIPFFT_NO_WORKSPACE"}
