package engine

import (
	"code.google.com/p/mx3/cuda"
	"github.com/barnex/cuda5/cu"
	"log"
	"unsafe"
)

type symmparam struct {
	lut [MAXREG * (MAXREG + 1) / 2]float32 // look-up table source
	gpu cuda.SymmLUT                       // gpu copy of lut, lazily transferred when needed
	ok  bool                               // gpu cache up-to date with lut source
}

//func (p *symmparam) setRegion(region int, v float64) {
//	p.setBetween(region, region, v)
//}

func (p *symmparam) SetInterRegion(r1, r2 int, v float64) {
	p.lut[symmidx(r1, r2)] = float32(v)
	p.ok = false
}

func (p *symmparam) SetUniform(v float64) {
	for i := range p.lut {
		p.lut[i] = float32(v)
	}
	p.ok = false
}

// index in symmetric matrix where only one half is stored
func symmidx(i, j int) int {
	if i <= j {
		return i*(i+1)/2 + j
	} else {
		return j*(j+1)/2 + i
	}
}

func (p *symmparam) getInter(r1, r2 int) float64 {
	return float64(p.lut[symmidx(r1, r2)])
}

//func (p *param) getUniform() []float64 {
//	v := make([]float64, p.NComp())
//	for c := range v {
//		x := p.lut[c][1]
//		for r := 2; r < MAXREG; r++ {
//			if p.lut[c][r] != x {
//				log.Panicf("%v is not uniform, need to specify a region (%v.GetRegion(x))", p.name, p.name)
//			}
//		}
//		v[c] = float64(x)
//	}
//	return v
//}

// Get returns the space-dependent parameter as a slice of floats, so it can be output.
//func (p *symmparam) Get() (*data.Slice, bool) {
//	s := data.NewSlice(1, p.Mesh())
//	l := s.Host()[0]
//
//	for i, r := range regions.cpu {
//		v := p.getInter(int(r), int(r))
//		l[i] = float32(v)
//	}
//	return s, false
//}

// Get a GPU mirror of the look-up table.
// Copies to GPU first only if needed.
func (p *symmparam) Gpu() cuda.SymmLUT {
	if p.ok {
		return p.gpu
	}
	p.upload()
	return p.gpu
}

// XYZ swap here
func (p *symmparam) upload() {
	if p.gpu == nil { // alloc only when needed, allows param use in init()
		p.gpu = cuda.SymmLUT(cuda.MemAlloc(int64(len(p.lut)) * cu.SIZEOF_FLOAT32))
	}
	log.Println("upload SymmLUT", p.lut[:20], "...")
	cu.MemcpyHtoD(cu.DevicePtr(p.gpu), unsafe.Pointer(&p.lut[0]), cu.SIZEOF_FLOAT32*int64(len(p.lut)))
	p.ok = true
}
