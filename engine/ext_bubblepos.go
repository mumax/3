package engine

import (
	"math"
)

var (
	BubblePos      = NewVectorValue("ext_bubblepos", "m", "Bubble core position", bubblePos)
	BubbleDist     = NewScalarValue("ext_bubbledist", "m", "Bubble traveled distance", bubbleDist)
	BubbleSpeed    = NewScalarValue("ext_bubblespeed", "m/s", "Bubble velocity", bubbleSpeed)
	BubbleMz       = 1.0
	BackGroundTilt = 0.25
)

func init() {
	DeclVar("ext_BubbleMz", &BubbleMz, "Center magnetization 1.0 or -1.0  (default = 1.0)")
	DeclVar("ext_BackGroundTilt", &BackGroundTilt, "Size of in-plane component of background magnetization. All values below this one are rounded down to perfectly out-of-plane to improve position calculation  (default = 0.25)")
}

func bubblePos() []float64 {
	m := M.Buffer()
	n := Mesh().Size()
	c := Mesh().CellSize()

	g := geometry.Gpu()
	var geo [][]float32
	if !g.IsNil() {
		geo = g.Comp(0).HostCopy().Scalars()[0] // geometry[Y, X]
	}

	mz := m.Comp(Z).HostCopy().Scalars()[0]

	posx, posy := 0., 0.

	if BubbleMz != -1.0 && BubbleMz != 1.0 {
		panic("ext_BubbleMz should be 1.0 or -1.0")
	}

	{
		var mag float64
		var magsum float64
		var weightedsumx float64
		var weightedsumy float64

		for ix := range mz[0] {
			for iy := range mz {
				mag = backgroundAdjust(mz[iy][ix]*float32(BubbleMz) + 1.) // 1/2 is divided out

				// weight cells according to geometry: 0 weight outside
				if !g.IsNil() {
					mag *= float64(geo[iy][ix])
				}

				magsum += mag
				weightedsumx += mag * float64(ix)
				weightedsumy += mag * float64(iy)
			}
		}
		posx = float64(weightedsumx / magsum)
		posy = float64(weightedsumy / magsum)
	}

	return []float64{(posx-float64(n[X]/2))*c[X] + GetShiftPos(), (posy-float64(n[Y]/2))*c[Y] + GetShiftYPos(), 0.}
}

var (
	prevBpos = [2]float64{-1e99, -1e99}
	bdist    = 0.0
)

func bubbleDist() float64 {
	pos := bubblePos()
	if prevBpos == [2]float64{-1e99, -1e99} {
		prevBpos = [2]float64{pos[X], pos[Y]}
		return 0
	}

	w := Mesh().WorldSize()
	dx := pos[X] - prevBpos[X]
	dy := pos[Y] - prevBpos[Y]
	prevBpos = [2]float64{pos[X], pos[Y]}

	// PBC wrap
	if dx > w[X]/2 {
		dx -= w[X]
	}
	if dx < -w[X]/2 {
		dx += w[X]
	}
	if dy > w[Y]/2 {
		dy -= w[Y]
	}
	if dy < -w[Y]/2 {
		dy += w[Y]
	}

	bdist += math.Sqrt(dx*dx + dy*dy)
	return bdist
}

var (
	prevBdist = 0.0
	prevBt    = -999.0
)

func bubbleSpeed() float64 {
	dist := bubbleDist()

	if prevBt < 0 {
		prevBdist = dist
		prevBt = Time
		return 0
	}

	v := (dist - prevBdist) / (Time - prevBt)

	prevBt = Time
	prevBdist = dist

	return v
}

func backgroundAdjust(arg float32) float64 {
	if float64(arg) < BackGroundTilt {
		return float64(0)
	}
	return float64(arg)
}
