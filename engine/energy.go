package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

func ExchangeEnergy() float64 {
	return -0.5 * mDotAdder(b_exch)
}

func DemagEnergy() float64 {
	return -0.5 * mDotSlice(b_demag.get())
}

func mDotAdder(a *adder) float64 {
	field := a.get_mustRecycle()
	defer cuda.RecycleBuffer(field)
	return mDotSlice(field)
}

func mDotSlice(field *data.Slice) float64 {
	util.Assert(vol == nil) // would need to include in dot
	c := mesh.CellSize()
	cellVolume := c[0] * c[1] * c[2]
	return (Msat() * cellVolume * float64(cuda.Dot(field, m.Slice)))
}
