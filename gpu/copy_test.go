package gpu

import (
	"code.google.com/p/mx3/nimble"
	"testing"
)

func TestCopy(t *testing.T) {
	LockCudaThread()

	cell := 1e-9
	mesh := nimble.NewMesh(2, 4, 8, cell, cell, cell)
	N := mesh.NCell()
	F := 100
	a := nimble.MakeChan1("a", "", mesh, nimble.CPUMemory, 0)

	up := NewUploader("b", "", a)
	down := NewDownloader("c", "", up.Output())
	c := down.Output()

	go func() {
		for f := 0; f < F; f++ {
			gpu := a.WriteNext(N).Host()
			for i := range gpu {
				gpu[i] = float32(i)
			}
			a.WriteDone()
		}
	}()

	C := c.NewReader()

	nimble.RunStack()

	for f := 0; f < F; f++ {
		gpu := C.ReadNext(N).Host()
		for i := range gpu {
			if gpu[i] != float32(i) {
				t.Error("expected:", float32(i), "got:", gpu[i])
			}
		}
		C.ReadDone()
	}
}
