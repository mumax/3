package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/nimble"
	"flag"
	"fmt"
	"github.com/barnex/cuda5/cu"
)

var cudaCtx cu.Context // gpu context to be used by all threads

func init() {
	flag.Parse()
	var flag uint
	switch *nimble.Flag_sched {
	default:
		panic("sched flag: expecting auto,spin,yield or sync: " + *nimble.Flag_sched)
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
	dev := cu.Device(*nimble.Flag_gpu)
	cudaCtx = cu.CtxCreate(flag, dev)
	M, m := dev.ComputeCapability()
	if *nimble.Flag_version {
		concurrent := dev.Attribute(cu.CONCURRENT_KERNELS)
		fmt.Print("CUDA ", float32(cu.Version())/1000, " ", dev.Name(), " (", (dev.TotalMem())/(1024*1024), "MB", ", compute", M, ".", m, ", concurrent:", concurrent == 1, ")\n")
	}
	if M < 2 {
		core.Log("Compute capability does not allow unified addressing.")
	}
	initStreamPool()
}

// cu.Init(), but error is fatal and does not dump stack.
func tryCuInit() {
	defer func() {
		err := recover()
		core.Fatal(err)
	}()
	cu.Init(0)
}
