package cuda

import (
	"code.google.com/p/mx3/data"
	"testing"
)

func TestStencil(t *testing.T) {
	const (
		N0 = 4
		N1 = 125
		N2 = 500
		c  = 1e-9
	)
	mesh := data.NewMesh(N0, N1, N2, c, c, c)

	stencil := &Stencil{[7]float32{1, 6, 7, 4, 5, 2, 3}}

	const x, y, z = 1, 121, 333

	in := NewUnifiedSlice(1, mesh)
	in.Scalars()[x][y][z] = 3
	out := NewUnifiedSlice(1, mesh)
	stencil.Exec(out, in)

	const want = 6
	got := out.Scalars()[x][y][z+1]
	if got != want {
		t.Fatalf("expected: %v, got: %v", want, got)
	}
}
