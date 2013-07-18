package cuda

import (
	"code.google.com/p/mx3/util"
	"flag"
	"github.com/barnex/cuda5/cu"
	"log"
	"runtime"
)

var (
	Flag_gpu      = flag.Int("gpu", 0, "specify GPU")
	Flag_sched    = flag.String("sched", "auto", "CUDA scheduling: auto|spin|yield|sync")
	Flag_pagelock = flag.Bool("pagelock", true, "enable CUDA memeory page-locking")
)

var (
	cudaCtx  cu.Context // gpu context to be used by all threads
	cudaCC   int        // compute capablity
	Version  float32
	DevName  string
	TotalMem int64
)

func Init() {
	if cudaCtx != 0 {
		return // already inited
	}
	var flag uint
	switch *Flag_sched {
	default:
		panic("sched flag: expecting auto,spin,yield or sync: " + *Flag_sched)
	case "auto":
		flag = cu.CTX_SCHED_AUTO
	case "spin":
		flag = cu.CTX_SCHED_SPIN
	case "yield":
		flag = cu.CTX_SCHED_YIELD
	case "sync":
		flag = cu.CTX_BLOCKING_SYNC
	}
	tryCuInit()
	dev := cu.Device(*Flag_gpu)
	cudaCtx = cu.CtxCreate(flag, dev)
	M, m := dev.ComputeCapability()
	concurrent := dev.Attribute(cu.CONCURRENT_KERNELS)

	Version = float32(cu.Version()) / 1000
	DevName = dev.Name()
	TotalMem = dev.TotalMem()

	log.Print("CUDA ", Version, " ",
		DevName, "(", (TotalMem)/(1024*1024), "MB) ",
		"compute ", M, ".", m, " concurrent: ", concurrent == 1, "\n")
	if M < 2 {
		log.Fatalln("GPU has insufficient compute capability, need 2.0 or higher.")
	}
	cudaCC = 10*M + m
	initStreampool()
}

// cu.Init(), but error is fatal and does not dump stack.
func tryCuInit() {
	defer func() {
		err := recover()
		util.FatalErr(err, "initialize GPU:")
	}()
	cu.Init(0)
}

// LockCudaThread locks the current goroutine to an OS thread
// and sets the CUDA context for that thread. To be called by
// every fresh goroutine that will use CUDA.
func LockThread() {
	runtime.LockOSThread()
	cudaCtx.SetCurrent()
}
