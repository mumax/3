package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	//"log"
)

func init() {
	//DeclFunc("average", Average, "Average of space-dependent quantity over magnet volume")
}

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
	avg := make([]float64, s.NComp())
	for i := range avg {
		avg[i] = float64(cuda.Sum(s.Comp(i))) / float64(Mesh().NCell())
	}
	return avg
}

func sAverageMagnet(s *data.Slice) []float64 {
	if geometry.Gpu().IsNil() {
		return sAverageUniverse(s)
	} else {
		avg := make([]float64, s.NComp())
		for i := range avg {
			avg[i] = float64(cuda.Dot(s.Comp(i), geometry.Gpu())) / magnetNCell()
		}
		return avg
	}
}

func magnetNCell() float64 {
	if geometry.Gpu().IsNil() {
		return float64(Mesh().NCell())
	} else {
		return float64(cuda.Sum(geometry.Gpu()))
	}
}

// Take average of quantity, taking into account the volume in which it is defined. E.g.:
// 	Average(m)           // averges only inside magnet.
// 	Average(m.Region(1)) // averages only in region 1
//func Average(s Quantity) []float64 {
//	buf, recycle := s.Slice()
//	if recycle {
//		defer cuda.Recycle(buf)
//	}
//	if v, ok := s.(volumer); ok {
//		avg := averageVolume(buf, nil)
//		spacefill := v.volume()
//		for i := range avg {
//			avg[i] /= spacefill
//		}
//		return avg
//	} else {
//		return averageVolume(buf, geometry.Gpu())
//	}
//}
//
//// Average of quantity with explicitly given volume mask.
//func averageVolume(b, vol *data.Slice) []float64 {
//	nComp := b.NComp()
//	nCell := float64(b.Len())
//	avg := make([]float64, nComp)
//	for i := range avg {
//		if vol.IsNil() {
//			avg[i] = float64(cuda.Sum(b.Comp(i))) / nCell
//		} else {
//			avg[i] = float64(cuda.Dot(b.Comp(i), vol)) / (spaceFill() * nCell)
//		}
//	}
//	return avg
//}
//
//func averageRegion(s *data.Slice, regionvolume float64) []float64 {
//	log.Println("regionvolume", regionvolume)
//	nComp := s.NComp()
//	nCell := float64(s.Len())
//	avg := make([]float64, nComp)
//	vol := geometry.Gpu()
//	for i := range avg {
//		if vol.IsNil() {
//			avg[i] = float64(cuda.Sum(s.Comp(i))) / (nCell * regionvolume)
//		} else {
//			avg[i] = float64(cuda.Dot(s.Comp(i), vol)) / (nCell * regionvolume)
//		}
//	}
//	return avg
//}
//
//// TODO: remove, make explicit
//type volumer interface {
//	volume() float64 // normalized volume where quantity is defined (0..1)
//}
