package gpu

import (
	"github.com/barnex/cuda4/safe"
	"nimble-cube/core"
	"testing"
)

func TestCopy(t *testing.T) {
	LockCudaThread()

	N := 64
	F := 100
	a := make([]float32, N)
	b := safe.MakeFloat32s(N)
	c := make([]float32, N)

	mA := core.NewRWMutex(N)
	mB := core.NewRWMutex(N)
	mC := core.NewRWMutex(N)

	up := NewUploader(a, mA.NewReader(), b, mB)
	down := NewDownloader(b, mB.NewReader(), c, mC)

	go up.Run()
	go down.Run()

	go func(){
	for f:=0; f<F; f++{
	mA.WLock(0, N)
	for i := range a {
		a[i] = float32(i)
	}
	mA.WLock(0, 0)
	}
	}()

	
	for f:=0; f<F; f++{
	mC.NewReader().RLock(0, N)
	for i := range c {
		if c[i] != float32(i) {
			t.Error("expected:", float32(i), "got:", c[i])
		}
	}
	mC.NewReader().RLock(0, 0)
	}
	core.Log(c)
}
