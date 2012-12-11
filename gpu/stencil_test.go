package gpu

import (
	"code.google.com/p/mx3/nimble"
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
	inCh := nimble.MakeChan1("in", "", mesh, nimble.UnifiedMemory, 0)
	w := [7]float32{1, 6, 7, 4, 5, 2, 3}
	stencil := NewStencil("out", "", inCh, w)

	const x, y, z = 1, 121, 333
	in := inCh.UnsafeArray()
	in[x][y][z] = 3
	stencil.Exec()
	stencil.Exec()
	stencil.Exec()
	out := stencil.Output().UnsafeArray()
	const want = 6
	got := out[x][y][z+1]
	if got != want {
		t.Fatalf("expected: %v, got: %v", want, got)
	}
}
