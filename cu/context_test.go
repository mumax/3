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
