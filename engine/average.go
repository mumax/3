package engine

// Averaging of quantities over entire universe or just magnet.

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// average of quantity over universe
func qAverageUniverse(q Quantity) []float64 {
	s, recycle := q.Slice()
	if recycle {
		defer cuda.Recycle(s)
	}
	return sAverageUniverse(s)
}

// average of slice over universe
func sAverageUniverse(s *data.Slice) []float64 {
	nCell := float64(prod(s.Size()))
	avg := make([]float64, s.NComp())
	for i := range avg {
		avg[i] = float64(cuda.Sum(s.Comp(i))) / nCell
		checkNaN1(avg[i])
	}
	return avg
}

// average of slice over the magnet volume
func sAverageMagnet(s *data.Slice) []float64 {
	if geometry.Gpu().IsNil() {
		return sAverageUniverse(s)
	} else {
		avg := make([]float64, s.NComp())
		for i := range avg {
			avg[i] = float64(cuda.Dot(s.Comp(i), geometry.Gpu())) / magnetNCell()
			checkNaN1(avg[i])
		}
		return avg
	}
}

// number of cells in the magnet.
// not necessarily integer as cells can have fractional volume.
func magnetNCell() float64 {
	if geometry.Gpu().IsNil() {
		return float64(Mesh().NCell())
	} else {
		return float64(cuda.Sum(geometry.Gpu()))
	}
}
