package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

//// Returns the maximum norm of a vector field.
//// TODO: only for vectors
//// TODO: does not belong here
//func (b *buffered) MaxNorm() float64 {
//	return cuda.MaxVecNorm(b.buffer)
//}
//

type getIface interface {
	getGPU() (s *data.Slice, mustRecylce bool)
}

// average in userspace XYZ order
// does not yet take into account volume.
// pass volume parameter, possibly nil?
func average(s getIface) []float64 {
	b, recycle := s.getGPU()
	if recycle {
		defer cuda.RecycleBuffer(b)
	}
	nComp := b.NComp()
	avg := make([]float64, nComp)
	for i := range avg {
		I := swapIndex(i, nComp)
		avg[i] = float64(cuda.Sum(b.Comp(I))) / float64(b.Mesh().NCell())
	}
	return avg
}
