package cu

// This file implements the CUDA Graph API: recording a sequence of stream
// operations into a graph, instantiating it as an executable graph, and
// replaying it with a single driver call (cuGraphLaunch) instead of one
// cuLaunchKernel per operation.
//
// Note: graph capture (StreamBeginCapture/EndCapture) is not supported on
// the legacy default stream (Stream(0)). Use a stream created with
// StreamCreate for capture and replay.

//#include <cuda.h>
import "C"

import (
	"unsafe"
)

// CUDA graph: a recorded sequence of operations (kernel launches, memcopies, ...).
type Graph uintptr

// Executable instantiation of a Graph, ready to be launched repeatedly.
type GraphExec uintptr

// A single node (e.g. a kernel launch) within a Graph.
type GraphNode uintptr

// Controls how StreamBeginCapture interacts with capture sequences on other streams.
type StreamCaptureMode int

const (
	STREAM_CAPTURE_MODE_GLOBAL       StreamCaptureMode = C.CU_STREAM_CAPTURE_MODE_GLOBAL
	STREAM_CAPTURE_MODE_THREAD_LOCAL StreamCaptureMode = C.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
	STREAM_CAPTURE_MODE_RELAXED      StreamCaptureMode = C.CU_STREAM_CAPTURE_MODE_RELAXED
)

// Type of a graph node, as returned by GraphNode.GetType.
type GraphNodeType int

const (
	GRAPH_NODE_TYPE_KERNEL           GraphNodeType = C.CU_GRAPH_NODE_TYPE_KERNEL
	GRAPH_NODE_TYPE_MEMCPY           GraphNodeType = C.CU_GRAPH_NODE_TYPE_MEMCPY
	GRAPH_NODE_TYPE_MEMSET           GraphNodeType = C.CU_GRAPH_NODE_TYPE_MEMSET
	GRAPH_NODE_TYPE_HOST             GraphNodeType = C.CU_GRAPH_NODE_TYPE_HOST
	GRAPH_NODE_TYPE_GRAPH            GraphNodeType = C.CU_GRAPH_NODE_TYPE_GRAPH
	GRAPH_NODE_TYPE_EMPTY            GraphNodeType = C.CU_GRAPH_NODE_TYPE_EMPTY
	GRAPH_NODE_TYPE_WAIT_EVENT       GraphNodeType = C.CU_GRAPH_NODE_TYPE_WAIT_EVENT
	GRAPH_NODE_TYPE_EVENT_RECORD     GraphNodeType = C.CU_GRAPH_NODE_TYPE_EVENT_RECORD
	GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL GraphNodeType = C.CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL
	GRAPH_NODE_TYPE_EXT_SEMAS_WAIT   GraphNodeType = C.CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT
	GRAPH_NODE_TYPE_MEM_ALLOC        GraphNodeType = C.CU_GRAPH_NODE_TYPE_MEM_ALLOC
	GRAPH_NODE_TYPE_MEM_FREE         GraphNodeType = C.CU_GRAPH_NODE_TYPE_MEM_FREE
	GRAPH_NODE_TYPE_BATCH_MEM_OP     GraphNodeType = C.CU_GRAPH_NODE_TYPE_BATCH_MEM_OP
	GRAPH_NODE_TYPE_CONDITIONAL      GraphNodeType = C.CU_GRAPH_NODE_TYPE_CONDITIONAL
)

// Parameters of a kernel-launch graph node, mirroring CUDA_KERNEL_NODE_PARAMS.
//
// KernelParams follows the same convention as LaunchKernel: each element is
// a pointer to the corresponding kernel argument's value. It must be set
// when calling GraphKernelNodeSetParams or GraphExecKernelNodeSetParams.
// GraphKernelNodeGetParams leaves it nil: the driver does not report the
// number of parameters, so the caller is expected to already know the
// argument layout of the node it created.
type KernelNodeParams struct {
	Func           Function
	GridDimX       int
	GridDimY       int
	GridDimZ       int
	BlockDimX      int
	BlockDimY      int
	BlockDimZ      int
	SharedMemBytes int
	KernelParams   []unsafe.Pointer
}

// Begins graph capture on a stream. While capturing, operations enqueued on
// the stream are not executed but recorded into a graph (see EndCapture).
func StreamBeginCapture(stream Stream, mode StreamCaptureMode) {
	err := Result(C.cuStreamBeginCapture(C.CUstream(unsafe.Pointer(uintptr(stream))), C.CUstreamCaptureMode(mode)))
	if err != SUCCESS {
		panic(err)
	}
}

// Begins graph capture on this stream.
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

// Creates an empty graph.
func GraphCreate() Graph {
	var graph C.CUgraph
	err := Result(C.cuGraphCreate(&graph, C.uint(0))) // flags must be zero
	if err != SUCCESS {
		panic(err)
	}
	return Graph(uintptr(unsafe.Pointer(graph)))
}

// Destroys a graph, along with the nodes it contains.
func GraphDestroy(graph *Graph) {
	err := Result(C.cuGraphDestroy(C.CUgraph(unsafe.Pointer(uintptr(*graph)))))
	*graph = 0
	if err != SUCCESS {
		panic(err)
	}
}

// Destroys the graph, along with the nodes it contains.
func (graph *Graph) Destroy() {
	GraphDestroy(graph)
}

// Returns the nodes that make up a graph.
func GraphGetNodes(graph Graph) []GraphNode {
	var n C.size_t
	err := Result(C.cuGraphGetNodes(C.CUgraph(unsafe.Pointer(uintptr(graph))), nil, &n))
	if err != SUCCESS {
		panic(err)
	}
	if n == 0 {
		return nil
	}
	cnodes := make([]C.CUgraphNode, int(n))
	err = Result(C.cuGraphGetNodes(C.CUgraph(unsafe.Pointer(uintptr(graph))), &cnodes[0], &n))
	if err != SUCCESS {
		panic(err)
	}
	nodes := make([]GraphNode, int(n))
	for i, c := range cnodes[:n] {
		nodes[i] = GraphNode(uintptr(unsafe.Pointer(c)))
	}
	return nodes
}

// Returns the nodes that make up this graph.
func (graph Graph) GetNodes() []GraphNode {
	return GraphGetNodes(graph)
}

// Returns the type of a graph node.
func GraphNodeGetType(node GraphNode) GraphNodeType {
	var t C.CUgraphNodeType
	err := Result(C.cuGraphNodeGetType(C.CUgraphNode(unsafe.Pointer(uintptr(node))), &t))
	if err != SUCCESS {
		panic(err)
	}
	return GraphNodeType(t)
}

// Returns the type of this graph node.
func (node GraphNode) GetType() GraphNodeType {
	return GraphNodeGetType(node)
}

// Returns the parameters of a kernel-launch graph node. KernelParams is left nil; see KernelNodeParams.
func GraphKernelNodeGetParams(node GraphNode) KernelNodeParams {
	var cparams C.CUDA_KERNEL_NODE_PARAMS
	err := Result(C.cuGraphKernelNodeGetParams(C.CUgraphNode(unsafe.Pointer(uintptr(node))), &cparams))
	if err != SUCCESS {
		panic(err)
	}
	return KernelNodeParams{
		Func:           Function(uintptr(unsafe.Pointer(cparams._func))),
		GridDimX:       int(cparams.gridDimX),
		GridDimY:       int(cparams.gridDimY),
		GridDimZ:       int(cparams.gridDimZ),
		BlockDimX:      int(cparams.blockDimX),
		BlockDimY:      int(cparams.blockDimY),
		BlockDimZ:      int(cparams.blockDimZ),
		SharedMemBytes: int(cparams.sharedMemBytes),
	}
}

// Returns the parameters of this kernel-launch graph node. KernelParams is left nil; see KernelNodeParams.
func (node GraphNode) KernelNodeGetParams() KernelNodeParams {
	return GraphKernelNodeGetParams(node)
}

// Sets the parameters of a kernel-launch graph node (the graph must not yet be instantiated).
func GraphKernelNodeSetParams(node GraphNode, p KernelNodeParams) {
	cparams, argv, argp := newCKernelNodeParams(&p)
	if argv != nil {
		defer C.free(argv)
		defer C.free(argp)
	}
	err := Result(C.cuGraphKernelNodeSetParams(C.CUgraphNode(unsafe.Pointer(uintptr(node))), &cparams))
	if err != SUCCESS {
		panic(err)
	}
}

// Sets the parameters of this kernel-launch graph node (the graph must not yet be instantiated).
func (node GraphNode) KernelNodeSetParams(p KernelNodeParams) {
	GraphKernelNodeSetParams(node, p)
}

// Instantiates a graph as an executable graph, ready to be launched
// repeatedly with GraphLaunch (and updated in place with
// GraphExecKernelNodeSetParams without re-instantiating).
func GraphInstantiate(graph Graph) GraphExec {
	var exec C.CUgraphExec
	err := Result(C.cuGraphInstantiate(&exec, C.CUgraph(unsafe.Pointer(uintptr(graph))), C.ulonglong(0)))
	if err != SUCCESS {
		panic(err)
	}
	return GraphExec(uintptr(unsafe.Pointer(exec)))
}

// Instantiates this graph as an executable graph. See GraphInstantiate.
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

// Updates the kernel-launch parameters of a node within an already
// instantiated executable graph, without re-instantiating it. Used to
// update per-step scalar arguments (e.g. dt, Time) of a captured kernel
// launch before replaying the graph with GraphLaunch.
func GraphExecKernelNodeSetParams(exec GraphExec, node GraphNode, p KernelNodeParams) {
	cparams, argv, argp := newCKernelNodeParams(&p)
	if argv != nil {
		defer C.free(argv)
		defer C.free(argp)
	}
	err := Result(C.cuGraphExecKernelNodeSetParams(C.CUgraphExec(unsafe.Pointer(uintptr(exec))), C.CUgraphNode(unsafe.Pointer(uintptr(node))), &cparams))
	if err != SUCCESS {
		panic(err)
	}
}

// Updates the kernel-launch parameters of a node within this executable graph. See GraphExecKernelNodeSetParams.
func (exec GraphExec) KernelNodeSetParams(node GraphNode, p KernelNodeParams) {
	GraphExecKernelNodeSetParams(exec, node, p)
}

// Builds a CUDA_KERNEL_NODE_PARAMS from p. If p.KernelParams is non-empty,
// argv and argp are C-allocated buffers (using the same double-copy as
// LaunchKernel, since a cgo argument cannot have a Go pointer to Go
// pointer) that the caller must free after the driver call returns.
func newCKernelNodeParams(p *KernelNodeParams) (cparams C.CUDA_KERNEL_NODE_PARAMS, argv, argp unsafe.Pointer) {
	if n := len(p.KernelParams); n > 0 {
		argv = C.malloc(C.size_t(n * pointerSize))
		argp = C.malloc(C.size_t(n * pointerSize))
		for i := range p.KernelParams {
			*((*unsafe.Pointer)(offset(argp, i))) = offset(argv, i)
			*((*uint64)(offset(argv, i))) = *((*uint64)(p.KernelParams[i]))
		}
	}
	cparams._func = C.CUfunction(unsafe.Pointer(uintptr(p.Func)))
	cparams.gridDimX = C.uint(p.GridDimX)
	cparams.gridDimY = C.uint(p.GridDimY)
	cparams.gridDimZ = C.uint(p.GridDimZ)
	cparams.blockDimX = C.uint(p.BlockDimX)
	cparams.blockDimY = C.uint(p.BlockDimY)
	cparams.blockDimZ = C.uint(p.BlockDimZ)
	cparams.sharedMemBytes = C.uint(p.SharedMemBytes)
	cparams.kernelParams = (*unsafe.Pointer)(argp)
	return cparams, argv, argp
}
