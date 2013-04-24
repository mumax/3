package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/util"
)

func ExchangeEnergy() float64 {
	util.Assert(vol == nil) // would need to include in dot
	bex := b_exch.get_mustRecycle()
	defer cuda.RecycleBuffer(bex)
	c := mesh.CellSize()
	cellVolume := c[0] * c[1] * c[2]
	return -0.5 * (Msat() * cellVolume * float64(cuda.Dot(bex, m.Slice)))
}
