//go:build hip

package cu

// This file provides access to HIP driver error statuses (type hipError_t).

//#include <hip/hip_runtime.h>
import "C"
import (
	"fmt"
)

// HIP error status.
// Error statuses are not returned by functions but checked and passed to
// panic() when not successful. If desired, they can be caught by
// recover().
type Result int

// Message string for the error
func (err Result) String() string {
	str, ok := errorString[err]
	if !ok {
		return "Unknown hipError_t: " + fmt.Sprint(int(err))
	}
	return str
}

const (
	SUCCESS                              Result = C.hipSuccess
	ERROR_INVALID_VALUE                  Result = C.hipErrorInvalidValue
	ERROR_OUT_OF_MEMORY                  Result = C.hipErrorOutOfMemory
	ERROR_NOT_INITIALIZED                Result = C.hipErrorNotInitialized
	ERROR_DEINITIALIZED                  Result = C.hipErrorDeinitialized
	ERROR_PROFILER_DISABLED              Result = C.hipErrorProfilerDisabled
	ERROR_PROFILER_NOT_INITIALIZED       Result = C.hipErrorProfilerNotInitialized
	ERROR_PROFILER_ALREADY_STARTED       Result = C.hipErrorProfilerAlreadyStarted
	ERROR_PROFILER_ALREADY_STOPPED       Result = C.hipErrorProfilerAlreadyStopped
	ERROR_NO_DEVICE                      Result = C.hipErrorNoDevice
	ERROR_INVALID_DEVICE                 Result = C.hipErrorInvalidDevice
	ERROR_INVALID_IMAGE                  Result = C.hipErrorInvalidImage
	ERROR_INVALID_CONTEXT                Result = C.hipErrorInvalidContext
	ERROR_CONTEXT_ALREADY_CURRENT        Result = C.hipErrorContextAlreadyCurrent
	ERROR_MAP_FAILED                     Result = C.hipErrorMapFailed
	ERROR_UNMAP_FAILED                   Result = C.hipErrorUnmapFailed
	ERROR_ARRAY_IS_MAPPED                Result = C.hipErrorArrayIsMapped
	ERROR_ALREADY_MAPPED                 Result = C.hipErrorAlreadyMapped
	ERROR_NO_BINARY_FOR_GPU              Result = C.hipErrorNoBinaryForGpu
	ERROR_ALREADY_ACQUIRED               Result = C.hipErrorAlreadyAcquired
	ERROR_NOT_MAPPED                     Result = C.hipErrorNotMapped
	ERROR_NOT_MAPPED_AS_ARRAY            Result = C.hipErrorNotMappedAsArray
	ERROR_NOT_MAPPED_AS_POINTER          Result = C.hipErrorNotMappedAsPointer
	ERROR_ECC_UNCORRECTABLE              Result = C.hipErrorECCNotCorrectable
	ERROR_UNSUPPORTED_LIMIT              Result = C.hipErrorUnsupportedLimit
	ERROR_CONTEXT_ALREADY_IN_USE         Result = C.hipErrorContextAlreadyInUse
	ERROR_INVALID_SOURCE                 Result = C.hipErrorInvalidSource
	ERROR_FILE_NOT_FOUND                 Result = C.hipErrorFileNotFound
	ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND Result = C.hipErrorSharedObjectSymbolNotFound
	ERROR_SHARED_OBJECT_INIT_FAILED      Result = C.hipErrorSharedObjectInitFailed
	ERROR_OPERATING_SYSTEM               Result = C.hipErrorOperatingSystem
	ERROR_INVALID_HANDLE                 Result = C.hipErrorInvalidHandle
	ERROR_NOT_FOUND                      Result = C.hipErrorNotFound
	ERROR_NOT_READY                      Result = C.hipErrorNotReady
	ERROR_LAUNCH_FAILED                  Result = C.hipErrorLaunchFailure
	ERROR_LAUNCH_OUT_OF_RESOURCES        Result = C.hipErrorLaunchOutOfResources
	ERROR_LAUNCH_TIMEOUT                 Result = C.hipErrorLaunchTimeOut
	ERROR_PEER_ACCESS_ALREADY_ENABLED    Result = C.hipErrorPeerAccessAlreadyEnabled
	ERROR_PEER_ACCESS_NOT_ENABLED        Result = C.hipErrorPeerAccessNotEnabled
	ERROR_PRIMARY_CONTEXT_ACTIVE         Result = C.hipErrorSetOnActiveProcess
	ERROR_CONTEXT_IS_DESTROYED           Result = C.hipErrorContextIsDestroyed
	ERROR_ASSERT                         Result = C.hipErrorAssert
	ERROR_HOST_MEMORY_ALREADY_REGISTERED Result = C.hipErrorHostMemoryAlreadyRegistered
	ERROR_HOST_MEMORY_NOT_REGISTERED     Result = C.hipErrorHostMemoryNotRegistered
	ERROR_ILLEGAL_ADDRESS                Result = C.hipErrorIllegalAddress
	ERROR_NOT_SUPPORTED                  Result = C.hipErrorNotSupported
	ERROR_UNKNOWN                        Result = C.hipErrorUnknown
)

// Map with error strings for Result error numbers
var errorString = map[Result]string{
	SUCCESS:                              "hipSuccess",
	ERROR_INVALID_VALUE:                  "hipErrorInvalidValue",
	ERROR_OUT_OF_MEMORY:                  "hipErrorOutOfMemory",
	ERROR_NOT_INITIALIZED:                "hipErrorNotInitialized",
	ERROR_DEINITIALIZED:                  "hipErrorDeinitialized",
	ERROR_PROFILER_DISABLED:              "hipErrorProfilerDisabled",
	ERROR_PROFILER_NOT_INITIALIZED:       "hipErrorProfilerNotInitialized",
	ERROR_PROFILER_ALREADY_STARTED:       "hipErrorProfilerAlreadyStarted",
	ERROR_PROFILER_ALREADY_STOPPED:       "hipErrorProfilerAlreadyStopped",
	ERROR_NO_DEVICE:                      "hipErrorNoDevice",
	ERROR_INVALID_DEVICE:                 "hipErrorInvalidDevice",
	ERROR_INVALID_IMAGE:                  "hipErrorInvalidImage",
	ERROR_INVALID_CONTEXT:                "hipErrorInvalidContext",
	ERROR_CONTEXT_ALREADY_CURRENT:        "hipErrorContextAlreadyCurrent",
	ERROR_MAP_FAILED:                     "hipErrorMapFailed",
	ERROR_UNMAP_FAILED:                   "hipErrorUnmapFailed",
	ERROR_ARRAY_IS_MAPPED:                "hipErrorArrayIsMapped",
	ERROR_ALREADY_MAPPED:                 "hipErrorAlreadyMapped",
	ERROR_NO_BINARY_FOR_GPU:              "hipErrorNoBinaryForGpu",
	ERROR_ALREADY_ACQUIRED:               "hipErrorAlreadyAcquired",
	ERROR_NOT_MAPPED:                     "hipErrorNotMapped",
	ERROR_NOT_MAPPED_AS_ARRAY:            "hipErrorNotMappedAsArray",
	ERROR_NOT_MAPPED_AS_POINTER:          "hipErrorNotMappedAsPointer",
	ERROR_ECC_UNCORRECTABLE:              "hipErrorECCNotCorrectable",
	ERROR_UNSUPPORTED_LIMIT:              "hipErrorUnsupportedLimit",
	ERROR_CONTEXT_ALREADY_IN_USE:         "hipErrorContextAlreadyInUse",
	ERROR_INVALID_SOURCE:                 "hipErrorInvalidSource",
	ERROR_FILE_NOT_FOUND:                 "hipErrorFileNotFound",
	ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: "hipErrorSharedObjectSymbolNotFound",
	ERROR_SHARED_OBJECT_INIT_FAILED:      "hipErrorSharedObjectInitFailed",
	ERROR_OPERATING_SYSTEM:               "hipErrorOperatingSystem",
	ERROR_INVALID_HANDLE:                 "hipErrorInvalidHandle",
	ERROR_NOT_FOUND:                      "hipErrorNotFound",
	ERROR_NOT_READY:                      "hipErrorNotReady",
	ERROR_LAUNCH_FAILED:                  "hipErrorLaunchFailure",
	ERROR_LAUNCH_OUT_OF_RESOURCES:        "hipErrorLaunchOutOfResources",
	ERROR_LAUNCH_TIMEOUT:                 "hipErrorLaunchTimeOut",
	ERROR_PEER_ACCESS_ALREADY_ENABLED:    "hipErrorPeerAccessAlreadyEnabled",
	ERROR_PEER_ACCESS_NOT_ENABLED:        "hipErrorPeerAccessNotEnabled",
	ERROR_PRIMARY_CONTEXT_ACTIVE:         "hipErrorSetOnActiveProcess",
	ERROR_CONTEXT_IS_DESTROYED:           "hipErrorContextIsDestroyed",
	ERROR_ASSERT:                         "hipErrorAssert",
	ERROR_HOST_MEMORY_ALREADY_REGISTERED: "hipErrorHostMemoryAlreadyRegistered",
	ERROR_HOST_MEMORY_NOT_REGISTERED:     "hipErrorHostMemoryNotRegistered",
	ERROR_ILLEGAL_ADDRESS:                "hipErrorIllegalAddress",
	ERROR_NOT_SUPPORTED:                  "hipErrorNotSupported",
	ERROR_UNKNOWN:                        "hipErrorUnknown"}
