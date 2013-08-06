package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
	"log"
	"unsafe"
)

// param stores a space-dependent material parameter,
// by keeping a look-up table mapping region index to float value.
// implementation allows arbitrary number of components,
// narrowed down to 1 or 3 in ScalarParam and VectorParam
type param struct {
	lut         [][MAXREG]float32 // look-up table source
	gpu         cuda.LUTPtrs      // gpu copy of lut, lazily transferred when needed
	ok          bool              // gpu cache up-to date with lut source
	zero        bool              // are all values zero (then we may skip corresponding kernel)
	post_update func(region int)  // called after region value changed, e.g., to update dependent params
	autosave                      // allow it to be saved
}

// constructor
func newParam(nComp int, name, unit string, post_update func(int)) param {
	return param{make([][MAXREG]float32, nComp), make(cuda.LUTPtrs, nComp), false, true, post_update, newAutosave(nComp, name, unit, nil)}
}

func (p *param) setRegion(region int, v ...float64) {
	util.Argument(len(v) == p.NComp()) // note: also panics if param not initialized (nComp = 0)

	v0 := true
	for c := range v {
		p.lut[c][region] = float32(v[c])
		if v[c] != 0 {
			v0 = false
		}
	}
	p.zero = false

	if v0 {
		p.zero = true
		for c := range p.lut {
			for i := range p.lut[c] {
				if p.lut[c][i] != 0 {
					p.zero = false
					c = len(p.lut) // break outer loop as well
					break
				}
			}
		}
	}

	p.ok = false
	if p.post_update != nil {
		p.post_update(region)
	}
}

func (p *param) getRegion(region int) []float64 {
	v := make([]float64, p.NComp())
	for c := range v {
		v[c] = float64(p.lut[c][region])
	}
	return v
}

// set in all regions except 0
func (p *param) setUniform(v ...float64) {
	for r := 1; r < MAXREG; r++ {
		p.setRegion(r, v...)
	}
}

func (p *param) getUniform() []float64 {
	v := make([]float64, p.NComp())
	for c := range v {
		x := p.lut[c][1]
		for r := 2; r < MAXREG; r++ {
			if p.lut[c][r] != x {
				log.Panicf("%v is not uniform, need to specify a region (%v.GetRegion(x))", p.name, p.name)
			}
		}
		v[c] = float64(x)
	}
	return v
}

func (p *param) GetVec() []float64 { return p.getUniform() }

// check if region is OK for use.
func checkRegion(region int) {
	if region == 0 {
		log.Fatal("cannot set parameters in region 0 (vacuum)")
	}
	if !regions.defined[region] {
		log.Panic("region ", region, " has not yet been defined by DefRegion()")
	}
}

func (p *param) GetGPU() (*data.Slice, bool) {
	p.upload()
	b := cuda.GetBuffer(p.NComp(), p.Mesh())
	for c := 0; c < p.NComp(); c++ {
		cuda.RegionDecode(b.Comp(c), cuda.LUTPtr(p.gpu[c]), regions.Gpu())
	}
	return b, true
}

func (p *param) Get() (*data.Slice, bool) {
	return p.GetGPU()
}

// Get a GPU mirror of the look-up table.
// Copies to GPU first only if needed.
func (p *param) Gpu() cuda.LUTPtrs {
	if p.ok {
		return p.gpu
	}
	p.upload()
	return p.gpu
}

// XYZ swap here
func (p *param) upload() {
	if p.gpu[0] == nil { // alloc only when needed, allows param use in init()
		for i := range p.gpu {
			p.gpu[i] = cuda.MemAlloc(MAXREG * cu.SIZEOF_FLOAT32)
		}
	}
	for c2 := range p.gpu {
		c := util.SwapIndex(c2, p.NComp())
		cu.MemcpyHtoD(cu.DevicePtr(p.gpu[c]), unsafe.Pointer(&p.lut[c2][0]), cu.SIZEOF_FLOAT32*MAXREG)
	}
	p.ok = true
}

func (p *adderQuant) Save() {
	save(p)
}

func (p *adderQuant) SaveAs(fname string) {
	saveAs(p, fname)
}
