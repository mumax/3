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

	up := NewUploader(a.MakeRChan(), b)
	down := NewDownloader(b.MakeRChan(), c)

	go up.Run()
	go down.Run()

	go func() {
		for f := 0; f < F; f++ {
			list := a.WriteNext(N)
			for i := range list {
				list[i] = float32(i)
			}
			a.WriteDone()
		}
	}()

	C := c.MakeRChan()
	for f := 0; f < F; f++ {
		list := C.ReadNext(N)
		for i := range list {
			if list[i] != float32(i) {
				t.Error("expected:", float32(i), "got:", list[i])
			}
		}
		C.ReadDone()
	}
	//core.Log(C.Uns)
}
