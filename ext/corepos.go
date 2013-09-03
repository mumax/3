package ext

import (
	"code.google.com/p/mx3/engine"
	"code.google.com/p/mx3/util"
)

func init() {
	engine.DeclROnly("ext_CorePos", &CorePos, "Vortex core position")
}

var CorePos = engine.NewGetVector("corepos", "m", corePos)

func corePos() []float64 {

	m, _ := engine.M.Get()
	util.Argument(m.Mesh().Size()[0] == 1) // 2D sim only
	mz := m.Comp(0).HostCopy().Scalars()[0]

	max := float32(-1.0)
	maxX, maxY := 0, 0
	for y := 1; y < len(mz)-1; y++ { // Avoid the boundaries so the neighbor interpolation can't go out of bounds.
		for x := 1; x < len(mz[y])-1; x++ {
			m := abs(mz[y][x])
			if m > max {
				maxX, maxY = x, y
				max = m
			}
		}
	}

	pos := make([]float64, 3)
	pos[0] = float64(maxX) + interpolate_maxpos(max, -1., abs(mz[maxY][maxX-1]), 1., abs(mz[maxY][maxX+1])) - float64(len(mz[1]))/2 + 0.5
	pos[1] = float64(maxY) + interpolate_maxpos(max, -1., abs(mz[maxY-1][maxX]), 1., abs(mz[maxY+1][maxX])) - float64(len(mz[0]))/2 + 0.5

	pos[0] *= engine.Mesh().CellSize()[2]
	pos[1] *= engine.Mesh().CellSize()[1]

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
