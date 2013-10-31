package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"log"
)

func init() {
	DeclFunc("average", Average, "Average of space-dependent quantity over magnet volume")
}

func Average(s Slicer) []float64 {
	buf, recycle := s.Slice()
	if recycle {
		defer cuda.Recycle(buf)
	}
	if v, ok := s.(volumer); ok {
		avg := avg(buf, nil)
		log.Println(s.Name(), "volume", v.volume())
		spacefill := v.volume()
		for i := range avg {
			avg[i] /= spacefill
		}
		return avg
	} else {
		return avg(buf, vol())
	}
}

func avg(b, vol *data.Slice) []float64 {
	nComp := b.NComp()
	nCell := float64(b.Mesh().NCell())
	avg := make([]float64, nComp)
	for i := range avg {
		I := i // util.SwapIndex(i, nComp)
		if vol.IsNil() {
			avg[i] = float64(cuda.Sum(b.Comp(I))) / nCell
		} else {
			avg[i] = float64(cuda.Dot(b.Comp(I), vol)) / (spaceFill() * nCell)
		}
	}
	return avg
}

type volumer interface {
	volume() float64
}
