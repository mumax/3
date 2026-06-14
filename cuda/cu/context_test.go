package cu

import (
	"fmt"
	"testing"
)

func TestContext(t *testing.T) {
	fmt.Println("CtxCreate")
	ctx := CtxCreate(CTX_SCHED_AUTO, 0)
	fmt.Println("CtxSetCurrent")
	CtxSetCurrent(ctx)
	fmt.Println("CtxGetApiVersion:", ctx.ApiVersion())
	fmt.Println("CtxGetDevice:", CtxGetDevice())
	(&ctx).Destroy()
}

func BenchmarkGetContext(b *testing.B) {
	b.StopTimer()
	ctx := CtxCreate(CTX_SCHED_AUTO, 0)
	CtxSetCurrent(ctx)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		CtxGetCurrent()
	}
}

func BenchmarkSetContext(b *testing.B) {
	b.StopTimer()
	ctx := CtxCreate(CTX_SCHED_AUTO, 0)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		ctx.SetCurrent()
	}
}
