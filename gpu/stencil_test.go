package gpu

import (
	"code.google.com/p/nimble-cube/nimble"
	"testing"
)

func TestStencil(t *testing.T) {
	const (
		N0 = 4
		N1 = 125
		N2 = 500
		c  = 1e-9
	)

	mesh := nimble.NewMesh(N0, N1, N2, c, c, c)
	in := nimble.MakeChan1("in", "", mesh, nimble.UnifiedMemory, 0)	
	w := [7]float32{1}
	stencil := NewStencil("out", "", in, w)
	stencil.Exec()
}
