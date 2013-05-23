package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"path"
)

type GetSlice interface {
	GetSlice() (s *data.Slice, mustRecylce bool)
}

// average in userspace XYZ order
// does not yet take into account volume.
// pass volume parameter, possibly nil?
func average(s GetSlice) []float64 {
	b, recycle := s.GetSlice()
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

// Save once, with given file name.
func saveAs(s GetSlice, fname string) {
	if !path.IsAbs(fname) {
		fname = OD + fname
	}
	buffer, recylce := s.GetSlice()
	if recylce {
		defer cuda.RecycleBuffer(buffer)
	}
	goSaveCopy(fname, buffer, Time)
}
