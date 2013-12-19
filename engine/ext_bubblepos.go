package engine

import (
	"math"
)

var (
	BubblePos   = NewGetVector("ext_bubblepos", "m", "Bubble core position", bubblePos)
	BubbleDist  = NewGetScalar("ext_bubbledist", "m", "Bubble traveled distance", bubbleDist)
	BubbleSpeed = NewGetScalar("ext_bubblespeed", "m/s", "Bubble velocity", bubbleSpeed)
)

func bubblePos() []float64 {
	m, _ := M.Slice()
	mz := m.Comp(Z).HostCopy().Scalars()[0]

	posx, posy := 0, 0

	{
		max := float32(-1e32)
		for iy := range mz {
			var sum float32
			for ix := range mz[iy] {
				sum += mz[iy][ix]
			}
			if sum > max {
				posy = iy
				max = sum
			}
		}
	}

	{
		max := float32(-1e32)
		for ix := range mz[0] {
			var sum float32
			for iy := range mz {
				sum += mz[iy][ix]
			}
			if sum > max {
				posx = ix
				max = sum
			}
		}
	}

	c := Mesh().CellSize()
	n := Mesh().Size()
	return []float64{float64(posx-n[X]/2)*c[X] + GetShiftPos(), float64(posy-n[Y]/2) * c[Y], 0}
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
