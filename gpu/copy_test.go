package gpu

//import (
//	"code.google.com/p/nimble-cube/nimble"
//	"testing"
//)
//
//func TestCopy(t *testing.T) {
//	LockCudaThread()
//
//	cell := 1e-9
//	mesh := nimble.NewMesh(2, 4, 8, cell, cell, cell)
//	N := mesh.NCell()
//	F := 100
//	a := nimble.MakeChan1("a", "", mesh)
//	b := MakeChan1("b", "", mesh)
//	c := nimble.MakeChan1("c", "", mesh)
//
//	up := NewUploader(a.NewReader(), b)
//	down := NewDownloader(b.NewReader(), c)
//
//	go up.Run()
//	go down.Run()
//
//	go func() {
//		for f := 0; f < F; f++ {
//			gpu := a.WriteNext(N)
//			for i := range gpu {
//				gpu[i] = float32(i)
//			}
//			a.WriteDone()
//		}
//	}()
//
//	C := c.NewReader()
//	for f := 0; f < F; f++ {
//		gpu := C.ReadNext(N)
//		for i := range gpu {
//			if gpu[i] != float32(i) {
//				t.Error("expected:", float32(i), "got:", gpu[i])
//			}
//		}
//		C.ReadDone()
//	}
//}
