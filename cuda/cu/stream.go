package cu

// This file implements CUDA streams

//#include <cuda.h>
import "C"
import "unsafe"

// CUDA stream.
type Stream uintptr

// Creates an asynchronous stream
func StreamCreate() Stream {
	var stream C.CUstream
	err := Result(C.cuStreamCreate(&stream, C.uint(0))) // flags has to be zero
	if err != SUCCESS {
		panic(err)
	}
	return Stream(uintptr(unsafe.Pointer(stream)))
}

// Destroys the asynchronous stream
func (stream *Stream) Destroy() {
	str := *stream
	err := Result(C.cuStreamDestroy(C.CUstream(unsafe.Pointer(uintptr(str)))))
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
	err := Result(C.cuStreamSynchronize(C.CUstream(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Returns Success if all operations have completed, ErrorNotReady otherwise
func (stream Stream) Query() Result {
	return Result(C.cuStreamQuery(C.CUstream(unsafe.Pointer(uintptr(stream)))))
}

// Returns Success if all operations have completed, ErrorNotReady otherwise
func StreamQuery(stream Stream) Result {
	return stream.Query()
}

// Blocks until the stream has completed.
func StreamSynchronize(stream Stream) {
	stream.Synchronize()
}
