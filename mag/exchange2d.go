package mag

//import (
//	"nimble-cube/core"
//)
//
//type Exchange2D struct {
//	m   core.RChan3
//	hex core.Chan3
//	*core.Mesh
//	aex_reduced float64
//}
//
//func (e *Exchange2D) Run() {
//	N := core.Prod(e.m.Size())
//	bl := core.BlockLen(e.m.Size())
//	bs := core.BlockSize(e.m.Size())
//
//	for {
//
//		Hex := core.Reshape3(e.hex.UnsafeArray(), e.hex.Size())[0] // 2D
//
//		off := 0
//		for I := 0; I < N; I += bl {
//			m := e.m.ReadNext(bl)
//			M := core.Reshape3(m, bs)[0] // 2D slice
//			// Write after read (sorry kc)
//			if I == 0 {
//				e.hex.WriteNext(N)
//			}
//
//			for i := range M {
//				for j := range M[i] {
//
//				}
//			}
//
//			off += len(M)
//		}
//		e.hex.WriteDone()
//	}
//}
//
//func NewExchange2D(m core.RChan3, hex core.Chan3, mesh *core.Mesh, aex_reduced float64) *Exchange2D {
//	return &Exchange2D{m, hex, mesh, aex_reduced}
//}
//
//// Naive implementation of 6-neighbor exchange field.
//// Aex in Tm² (exchange stiffness divided by Msat0).
//// Hex in Tesla.
////func exchange2d(m [3][][][]float32, Hex [3][][][]float32, cellsize [3]float64, aex_reduced float64) {
////	var (
////		facI = float32(aex_reduced / (cellsize[0] * cellsize[0]))
////		facJ = float32(aex_reduced / (cellsize[1] * cellsize[1]))
////		facK = float32(aex_reduced / (cellsize[2] * cellsize[2]))
////	)
////	N0, N1, N2 := len(m[0]), len(m[0][0]), len(m[0][0][0])
////
////}
