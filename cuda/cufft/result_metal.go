//go:build darwin && arm64
// +build darwin,arm64

package cufft

import "fmt"

type Result int

const (
	SUCCESS                   Result = 0
	INVALID_PLAN              Result = 1
	ALLOC_FAILED              Result = 2
	INVALID_TYPE              Result = 3
	INVALID_VALUE             Result = 4
	INTERNAL_ERROR            Result = 5
	EXEC_FAILED               Result = 6
	SETUP_FAILED              Result = 7
	INVALID_SIZE              Result = 8
	UNALIGNED_DATA            Result = 9
	INCOMPLETE_PARAMETER_LIST Result = 0xA
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

var resultString = map[Result]string{
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
	NO_WORKSPACE:              "CUFFT_NO_WORKSPACE",
}
