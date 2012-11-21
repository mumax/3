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
	inCh := nimble.MakeChan1("in", "", mesh, nimble.UnifiedMemory, 0)
	w := [7]float32{2}
	stencil := NewStencil("out", "", inCh, w)

	in := inCh.UnsafeData().Host()
	in[0] = 3
	stencil.Exec()
	out := stencil.Output().UnsafeData().Host()
	const want = 6
	got := out[0]
	if got != want {
		t.Fatalf("expected: %v, got: %v", want, got)
	}
}
