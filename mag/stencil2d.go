package mag

import (
	"nimble-cube/core"
)

type Stencil2D struct {
	in   core.RChan3
	out core.Chan3
	weights
}

func (e *Stencil2D) Run() {
	N := core.Prod(e.m.Size())
	bl := core.BlockLen(e.m.Size())
	nB := div(N , bl)
	bs := core.BlockSize(e.m.Size())
	bs2 := bs // size of two subsequent m blocks
	bs[1] *= 2 

	for {
		m := e.m.ReadNext(2*N)
		M := core.Reshape3(m, bs2)
		hex := e.hex.WriteNext(bl)
		Hex := core.Reshape3(hex, bs)[0] // 2D

		// top row
		for c := range Hex{
		for j:=0; j<len(Hex[c][0]); j++{
			
			Hex[c][0][j] = 
		}
		}

		for b := 0; b < nB-1; b++ {

	
			e.m.ReadDelta(N, N)
		}
	}
}

func NewStencil2D(m core.RChan3, hex core.Chan3, mesh *core.Mesh, aex_reduced float64) *Stencil2D {
	return &Stencil2D{m, hex, mesh, aex_reduced}
}

// Naive implementation of 6-neighbor exchange field.
// Aex in Tm² (exchange stiffness divided by Msat0).
// Hex in Tesla.
//func exchange2d(m [3][][][]float32, Hex [3][][][]float32, cellsize [3]float64, aex_reduced float64) {
//	var (
//		facI = float32(aex_reduced / (cellsize[0] * cellsize[0]))
//		facJ = float32(aex_reduced / (cellsize[1] * cellsize[1]))
//		facK = float32(aex_reduced / (cellsize[2] * cellsize[2]))
//	)
//	N0, N1, N2 := len(m[0]), len(m[0][0]), len(m[0][0][0])
//
//}

func div(a, b int) int{
	if a % b != 0{
		core.Panic(a, "%", b, "!=", 0)
	}
	return a / b
}
