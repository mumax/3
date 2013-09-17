package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

func init() {
	DeclFunc("average", Average, "Average of space-dependent quantity")
	DeclFunc("makeAvg", MakeAvg, "Make new quantity, average of argument")
	DeclFunc("makeAvgRegion", MakeAvgRegion, "Make new quantity, average of argument in region")
}

type reduced struct {
	doc
	calc func() []float64
}

func newReduced(nComp int, name, unit string, calc func() []float64) *reduced {
	return &reduced{Doc(nComp, name, unit), calc}
}

func (r *reduced) GetVec() []float64 {
	return r.calc()
}

func MakeAvgRegion(s Getter, region int) *reduced {
	return newReduced(s.NComp(), s.Name(), s.Unit(), func() []float64 {
		src, r := s.Get()
		if r {
			defer cuda.RecycleBuffer(src)
		}
		out := cuda.GetBuffer(s.NComp(), s.Mesh())
		defer cuda.RecycleBuffer(out)
		cuda.RegionSelect(out, src, regions.Gpu(), byte(region))
		return avg(out)
	})
}

func MakeAvg(s Getter) *reduced {
	return newReduced(s.NComp(), s.Name(), s.Unit(), func() []float64 {
		full := Average(s)
		for i := range full {
			full[i] /= spaceFill
		}
		return full
	})
}

// average in userspace XYZ order
// does not yet take into account volume.
// pass volume parameter, possibly nil?
func Average(s Getter) []float64 {
	b, recycle := s.Get()
	if recycle {
		defer cuda.RecycleBuffer(b)
	}
	return avg(b)
}

// userspace average
func avg(b *data.Slice) []float64 {
	nComp := b.NComp()
	avg := make([]float64, nComp)
	for i := range avg {
		I := util.SwapIndex(i, nComp)
		avg[i] = float64(cuda.Sum(b.Comp(I))) / float64(b.Mesh().NCell())
	}
	return avg
}
