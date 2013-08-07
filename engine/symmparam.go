package engine

import (
	"code.google.com/p/mx3/cuda"
	"github.com/barnex/cuda5/cu"
	"fmt"
	"unsafe"
)

type symmparam struct {
	lut [MAXREG * (MAXREG + 1) / 2]float32 // look-up table source
	gpu cuda.SymmLUT                       // gpu copy of lut, lazily transferred when needed
	ok  bool                               // gpu cache up-to date with lut source
}

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

// Index in symmetric matrix where only one half is stored.
// (!) Code duplicated in exchange.cu
func symmidx(i, j int) int {
	if j <= i {
		return i*(i+1)/2 + j
	} else {
		return j*(j+1)/2 + i
	}
}

func (p *symmparam) getInter(r1, r2 int) float64 {
	return float64(p.lut[symmidx(r1, r2)])
}

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
	fmt.Println("upload SymmLUT\n", p)
	cu.MemcpyHtoD(cu.DevicePtr(p.gpu), unsafe.Pointer(&p.lut[0]), cu.SIZEOF_FLOAT32*int64(len(p.lut)))
	p.ok = true
}

func(p*symmparam)String()string{
	str := ""
	for j := 0;  j < MAXREG; j++{
		for i:=0; i <= j; i++{
			str += fmt.Sprint(p.getInter(j, i), "\t")
		}
		str += "\n"
	}
	return str
}
