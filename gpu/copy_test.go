package gpu

import (
	"nimble-cube/core"
	"testing"
)

func TestCopy(t *testing.T) {
	LockCudaThread()

	size := [3]int{1, 4, 8}
	N := core.Prod(size)
	F := 100
	a := core.MakeChan(size)
	b := MakeChan(size)
	c := core.MakeChan(size)

	up := NewUploader(a.ReadOnly(), b)
	down := NewDownloader(b.ReadOnly(), c)

	go up.Run()
	go down.Run()

	go func() {
		for f := 0; f < F; f++ {
			a.WriteNext(N)
			for i := range a.List {
				a.List[i] = float32(i)
			}
			a.WriteDone()
		}
	}()

	C := c.ReadOnly()
	for f := 0; f < F; f++ {
		C.ReadNext(N)
		for i := range C.List {
			if C.List[i] != float32(i) {
				t.Error("expected:", float32(i), "got:", C.List[i])
			}
		}
		C.ReadDone()
	}
	core.Log(C.List)
}
