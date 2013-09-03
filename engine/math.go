package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/util"
)

// average in userspace XYZ order
// does not yet take into account volume.
// pass volume parameter, possibly nil?
func Average(s Getter) []float64 {
	b, recycle := s.Get()
	if recycle {
		defer cuda.RecycleBuffer(b)
	}
	nComp := b.NComp()
	avg := make([]float64, nComp)
	for i := range avg {
		I := util.SwapIndex(i, nComp)
		avg[i] = float64(cuda.Sum(b.Comp(I))) / float64(b.Mesh().NCell())
	}
	return avg
}

// Constructs a vector
func Vector(x, y, z float64) [3]float64 {
	return [3]float64{x, y, z}
}
