package safe

import (
	"github.com/barnex/cuda5/cu"
	"runtime"
)

func InitCuda() {
	runtime.LockOSThread()
	cu.Init(0)
	cu.CtxCreate(cu.CTX_SCHED_AUTO, 0).SetCurrent()
}
