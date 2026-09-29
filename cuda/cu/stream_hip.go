//go:build hip

package cu

// This file implements HIP streams

//#include <hip/hip_runtime.h>
import "C"
import "unsafe"

// HIP stream.
type Stream uintptr

// Creates an asynchronous stream
func StreamCreate() Stream {
	var stream C.hipStream_t
	err := Result(C.hipStreamCreate(&stream))
	if err != SUCCESS {
		panic(err)
	}
	return Stream(uintptr(unsafe.Pointer(stream)))
}

// Destroys the asynchronous stream
func (stream *Stream) Destroy() {
	str := *stream
	err := Result(C.hipStreamDestroy(C.hipStream_t(unsafe.Pointer(uintptr(str)))))
	*stream = 0
	if err != SUCCESS {
		panic(err)
	}
}

// Destroys an asynchronous stream
func StreamDestroy(stream *Stream) {
	stream.Destroy()
}

// Blocks until the stream has completed.
func (stream Stream) Synchronize() {
	err := Result(C.hipStreamSynchronize(C.hipStream_t(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Returns Success if all operations have completed, ErrorNotReady otherwise
func (stream Stream) Query() Result {
	return Result(C.hipStreamQuery(C.hipStream_t(unsafe.Pointer(uintptr(stream)))))
}

// Returns Success if all operations have completed, ErrorNotReady otherwise
func StreamQuery(stream Stream) Result {
	return stream.Query()
}

// Blocks until the stream has completed.
func StreamSynchronize(stream Stream) {
	stream.Synchronize()
}
