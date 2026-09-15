package cu

// [Requires CUDA >= 10.1]
//
// This file implements the CUDA Graph API: recording a sequence of stream
// operations into a graph, instantiating it as an executable graph, and
// replaying it with a single driver call (cuGraphLaunch) instead of one
// cuLaunchKernel per operation.
//
// Note: graph capture (Stream.BeginCapture/EndCapture) is not supported on
// the legacy default stream (Stream(0)). Use a stream created with
// StreamCreate for capture and replay.

/*
#include <cuda.h>

// CUDA 12.x redefined cuGraphInstantiate with different arguments. The old
// signature was renamed to cuGraphInstantiate_v2, but that did not yet exist
// before CUDA 12.0. Hence, we need the following if-statement, in which we
// define a custom C.mumaxGraphInstantiate covering all CUDA versions >=10.1.
#if CUDA_VERSION >= 12000
	static CUresult mumaxGraphInstantiate(CUgraphExec *exec, CUgraph graph) {
	    return cuGraphInstantiate(exec, graph, 0);
	}
#else
	static CUresult mumaxGraphInstantiate(CUgraphExec *exec, CUgraph graph) {
	    return cuGraphInstantiate(exec, graph, NULL, NULL, 0);
	}
#endif
*/
import "C"

import (
	"unsafe"
)

// CUDA graph: a recorded sequence of operations (kernel launches, memcopies, ...).
type Graph uintptr

// Executable instantiation of a Graph, ready to be launched repeatedly.
type GraphExec uintptr

// Controls how StreamBeginCapture interacts with capture sequences on other streams.
type StreamCaptureMode int

const (
	STREAM_CAPTURE_MODE_GLOBAL       StreamCaptureMode = C.CU_STREAM_CAPTURE_MODE_GLOBAL
	STREAM_CAPTURE_MODE_THREAD_LOCAL StreamCaptureMode = C.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
	STREAM_CAPTURE_MODE_RELAXED      StreamCaptureMode = C.CU_STREAM_CAPTURE_MODE_RELAXED
)

// Begins graph capture on a stream. While capturing, operations enqueued on
// the stream are not executed but recorded into a graph (see EndCapture).
func StreamBeginCapture(stream Stream, mode StreamCaptureMode) {
	err := Result(C.cuStreamBeginCapture(C.CUstream(unsafe.Pointer(uintptr(stream))), C.CUstreamCaptureMode(mode)))
	if err != SUCCESS {
		panic(err)
	}
}

// Begins graph capture on this stream. While capturing, operations enqueued on
// the stream are not executed but recorded into a graph (see EndCapture).
func (stream Stream) BeginCapture(mode StreamCaptureMode) {
	StreamBeginCapture(stream, mode)
}

// Ends graph capture on a stream, returning the captured graph.
func StreamEndCapture(stream Stream) Graph {
	var graph C.CUgraph
	err := Result(C.cuStreamEndCapture(C.CUstream(unsafe.Pointer(uintptr(stream))), &graph))
	if err != SUCCESS {
		panic(err)
	}
	return Graph(uintptr(unsafe.Pointer(graph)))
}

// Ends graph capture on this stream, returning the captured graph.
func (stream Stream) EndCapture() Graph {
	return StreamEndCapture(stream)
}

// Destroys a graph, along with the nodes it contains.
func GraphDestroy(graph *Graph) {
	err := Result(C.cuGraphDestroy(C.CUgraph(unsafe.Pointer(uintptr(*graph)))))
	*graph = 0
	if err != SUCCESS {
		panic(err)
	}
}

// Destroys this graph, along with the nodes it contains.
func (graph *Graph) Destroy() {
	GraphDestroy(graph)
}

// Instantiates a graph as an executable graph, ready to be launched
// repeatedly with GraphLaunch.
func GraphInstantiate(graph Graph) GraphExec {
	var exec C.CUgraphExec
	err := Result(C.mumaxGraphInstantiate(&exec, C.CUgraph(unsafe.Pointer(uintptr(graph)))))
	if err != SUCCESS {
		panic(err)
	}
	return GraphExec(uintptr(unsafe.Pointer(exec)))
}

// Instantiates this graph as an executable graph, ready to be launched
// repeatedly with GraphExec.Launch.
func (graph Graph) Instantiate() GraphExec {
	return GraphInstantiate(graph)
}

// Launches an executable graph on a stream.
func GraphLaunch(exec GraphExec, stream Stream) {
	err := Result(C.cuGraphLaunch(C.CUgraphExec(unsafe.Pointer(uintptr(exec))), C.CUstream(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Launches this executable graph on a stream.
func (exec GraphExec) Launch(stream Stream) {
	GraphLaunch(exec, stream)
}

// Destroys an executable graph.
func GraphExecDestroy(exec *GraphExec) {
	err := Result(C.cuGraphExecDestroy(C.CUgraphExec(unsafe.Pointer(uintptr(*exec)))))
	*exec = 0
	if err != SUCCESS {
		panic(err)
	}
}

// Destroys this executable graph.
func (exec *GraphExec) Destroy() {
	GraphExecDestroy(exec)
}
