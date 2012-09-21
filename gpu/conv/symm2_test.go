package conv

import (
	"nimble-cube/core"
	"nimble-cube/mag"
	"testing"
)

func TestSymm2(t *testing.T) {
	C := 1e-9
	mesh := core.NewMesh(N0, N1, N2, C, 2*C, 3*C)
	N := mesh.NCell()

	input := core.MakeVectors(mesh.GridSize())
	output := core.MakeVectors(mesh.GridSize())

	hinlock := [3]*core.RWMutex{core.NewRWMutex(N), core.NewRWMutex(N), core.NewRWMutex(N)}
	houtlock := [3]*core.RWMutex{core.NewRWMutex(N), core.NewRWMutex(N), core.NewRWMutex(N)}

	acc := 2
	kern := mag.BruteKernel(mesh.ZeroPadded(), acc)

	c := NewSymm2(mesh.GridSize(), mesh.ZeroPadded())

	up0 := NewUploader(input[0], hinlock[0], c.ioBuf[0], dlock[0])
}
