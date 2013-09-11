package ext

import (
	"code.google.com/p/mx3/engine"
)

var CorePos = engine.NewGetVector("ext_corepos", "m", "Vortex core position", corePos)

func corePos() []float64 {

	m, _ := engine.M.Get()
	mz3d := m.Comp(0).HostCopy().Scalars()

	s := m.Mesh().Size()
	Nx, Ny, Nz := s[2], s[1], s[0] // (xyz swap)

	max := float32(-1.0)
	var maxX, maxY, maxZ int

	for z := 0; z < Nz; z++ {
		mz := mz3d[z]
		// Avoid the boundaries so the neighbor interpolation can't go out of bounds.
		for y := 1; y < Ny-1; y++ {
			for x := 1; x < Nx-1; x++ {
				m := abs(mz[y][x])
				if m > max {
					maxX, maxY, maxZ = x, y, z
					max = m
				}
			}
		}
	}

	pos := make([]float64, 3)

	mz := mz3d[maxZ]
	pos[0] = float64(maxX) + interpolate_maxpos(max, -1., abs(mz[maxY][maxX-1]), 1., abs(mz[maxY][maxX+1])) - float64(len(mz[1]))/2 + 0.5
	pos[1] = float64(maxY) + interpolate_maxpos(max, -1., abs(mz[maxY-1][maxX]), 1., abs(mz[maxY+1][maxX])) - float64(len(mz[0]))/2 + 0.5
	pos[2] = float64(maxZ) - float64(Nz)/2 + 0.5

	pos[0] *= engine.Mesh().CellSize()[2] // (xyz swap)
	pos[1] *= engine.Mesh().CellSize()[1]
	pos[2] *= engine.Mesh().CellSize()[0]

	pos[0] += totalShift // add simulation window shift
	return pos
}

func interpolate_maxpos(f0, d1, f1, d2, f2 float32) float64 {
	b := (f2 - f1) / (d2 - d1)
	a := ((f2-f0)/d2 - (f0-f1)/(-d1)) / (d2 - d1)
	return float64(-b / (2 * a))
}

func abs(x float32) float32 {
	if x > 0 {
		return x
	} else {
		return -x
	}
}
