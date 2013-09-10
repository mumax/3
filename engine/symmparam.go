package engine

import (
	"code.google.com/p/mx3/cuda"
	"fmt"
	"github.com/barnex/cuda5/cu"
	"log"
	"math"
	"unsafe"
)

type symmparam struct {
	lut     [NREGION * (NREGION + 1) / 2]float32 // look-up table source
	gpu     cuda.SymmLUT                         // gpu copy of lut, lazily transferred when needed
	ok      bool                                 // gpu cache up-to date with lut source
	modtime float64
	// TODO: setmodtime: also clears OK flag, mainly in param
}

func (p *symmparam) init() {
	p.modtime = math.Inf(-1)
}

// Get a GPU mirror of the look-up table.
// Copies to GPU first only if needed.
func (p *symmparam) Gpu() cuda.SymmLUT {
	p.update()
	if !p.ok {
		p.upload()
	}
	return p.gpu
}

func (p *symmparam) update() {
	msat, tMsat := Msat.Cpu()
	aex, tAex := Aex.Cpu()

	if p.modtime < tMsat || p.modtime < tAex {
		for i := 0; i < regions.maxreg; i++ {
			lexi := 2e18 * safediv(aex[0][i], msat[0][i])
			for j := 0; j <= i; j++ {
				lexj := 2e18 * safediv(aex[0][j], msat[0][j])
				p.lut[symmidx(i, j)] = 2 / (1/lexi + 1/lexj)
			}
		}
		p.modtime = Time
		p.ok = false
	}
}

func safediv(a, b float32) float32 {
	if b == 0 {
		return 0
	} else {
		return a / b
	}
}

// XYZ swap here
func (p *symmparam) upload() {
	// alloc if  needed
	if p.gpu == nil {
		p.gpu = cuda.SymmLUT(cuda.MemAlloc(int64(len(p.lut)) * cu.SIZEOF_FLOAT32))
	}
	log.Println(" ******* upload Aex", p)
	cu.MemcpyHtoD(cu.DevicePtr(p.gpu), unsafe.Pointer(&p.lut[0]), cu.SIZEOF_FLOAT32*int64(len(p.lut)))
	p.ok = true
}

func (p *symmparam) SetInterRegion(r1, r2 int, val float64) {
	v := float32(val)
	p.lut[symmidx(r1, r2)] = v

	if r1 == r2 {
		r := r1
		for i := 0; i < NREGION; i++ {
			if p.lut[symmidx(i, i)] == v {
				p.lut[symmidx(r, i)] = v
			} else {
				p.lut[symmidx(r, i)] = 0
			}
		}
	}

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

func (p *symmparam) String() string {
	str := ""
	for j := 0; j < regions.maxreg; j++ {
		for i := 0; i <= j; i++ {
			str += fmt.Sprint(p.getInter(j, i), "\t")
		}
		str += "\n"
	}
	return str
}
