package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func init() {
	DeclFunc("average", Average, "Average of space-dependent quantity over magnet volume")
}

// average in userspace XYZ order
func Average(s Slicer) []float64 {
	b, recycle := s.Slice()
	if recycle {
		defer cuda.Recycle(b)
	}
	if s, ok := s.(volumer); ok {
		avg := avg(b, nil)
		spacefill := s.volume()
		for i := range avg {
			avg[i] /= spacefill
		}
		return avg
	} else {
		return avg(b, vol)
	}
}

// userspace average
func avg(b, vol *data.Slice) []float64 {
	nComp := b.NComp()
	nCell := float64(b.Mesh().NCell())
	avg := make([]float64, nComp)
	for i := range avg {
		I := util.SwapIndex(i, nComp)
		if vol.IsNil() {
			avg[i] = float64(cuda.Sum(b.Comp(I))) / nCell
		} else {
			avg[i] = float64(cuda.Dot(b.Comp(I), vol)) / (spaceFill * nCell)
		}
	}
	return avg
}

type volumer interface {
	volume() float64
}
