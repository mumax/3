// Package cuda provides GPU interaction
package cuda

import (
	"fmt"
	"log"
	"runtime"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/util"
)

var (
	DriverVersion int        // cuda driver version
	DevName       string     // GPU name
	TotalMem      int64      // total GPU memory
	GPUInfo       string     // Human-readable GPU description
	Synchronous   bool       // for debug: synchronize stream0 at every kernel launch
	cudaCtx       cu.Context // global CUDA context
	cudaCC        int        // compute capablity (used for fatbin)
)

// Locks to an OS thread and initializes CUDA for that thread.
func Init(gpu int) {
	if cudaCtx != 0 {
		return // needed for tests
	}

	runtime.LockOSThread()
	tryCuInit()
	dev := cu.Device(gpu)
	cudaCtx = cu.CtxCreate(cu.CTX_SCHED_YIELD, dev)
	cudaCtx.SetCurrent()

	M, m := dev.ComputeCapability()
	cudaCC = 10*M + m
	DriverVersion = cu.Version()
	DevName = dev.Name()
	TotalMem = dev.TotalMem()
	GPUInfo = fmt.Sprintf("%s(%dMB), CUDA Driver %d.%d, cc=%d.%d",
		DevName, (TotalMem)/(1024*1024), DriverVersion/1000, (DriverVersion%1000)/10, M, m)

	if M < 2 {
		log.Fatalln("GPU has insufficient compute capability, need 2.0 or higher.")
	}
	if Synchronous {
		log.Println("DEBUG: synchronized CUDA calls")
	}

	// test PTX load so that we can catch CUDA_ERROR_NO_BINARY_FOR_GPU early
	fatbinLoad(madd2_map, "madd2")
}

// cu.Init(), but error is fatal and does not dump stack.
func tryCuInit() {
	defer func() {
		err := recover()
		if err == cu.ERROR_UNKNOWN {
			log.Println("\n Try running: sudo nvidia-modprobe -u \n")
		}
		util.FatalErr(err)
	}()
	cu.Init(0)
}

// Global stream used for everything. Normally the NULL/legacy default
// stream (cu.Stream(0)), which all generated *_wrapper.go kernel launches
// reference directly via this package-level variable. CUDA Graph capture is
// not supported on the NULL stream, so capture/replay temporarily
// redirects stream0 to a dedicated stream (see EnterCaptureMode).
var stream0 = cu.Stream(0)

// Synchronize the global stream
// This is called before and after all memcopy operations between host and device.
func Sync() {
	stream0.Synchronize()
}

// Redirects stream0 to a freshly created stream suitable for CUDA Graph
// capture (cuStreamBeginCapture is not supported on the NULL stream), and
// returns that stream. Since every kernel launch references stream0
// directly, this single reassignment is enough to route all subsequent
// launches to the capture stream.
//
// FFT plans (e.g. the demag convolution's fwPlan/bwPlan) are bound to a
// stream once at creation time and do not follow this reassignment; callers
// must rebind those separately (see DemagConvolution.SetStream).
//
// Must be paired with a call to ExitCaptureMode.
func EnterCaptureMode() cu.Stream {
	captureStream := cu.StreamCreate()
	stream0 = captureStream
	return captureStream
}

// Restores stream0 to the NULL/legacy default stream and destroys the
// stream created by EnterCaptureMode.
func ExitCaptureMode(captureStream cu.Stream) {
	stream0 = cu.Stream(0)
	captureStream.Destroy()
}
