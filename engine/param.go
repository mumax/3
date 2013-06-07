package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
	"log"
	"unsafe"
)

const LUTSIZE = 256

// Param stores a space-dependent material parameter
// by keeping a look-up table mapping region index to float value.
type Param struct {
	lut         [][LUTSIZE]float32 // look-up table source
	gpu         cuda.LUTPtrs       // gpu copy of lut, lazily transferred when needed
	ok          bool               // gpu cache up-to date with lut source
	post_update func(region int)   // called after region value changed, e.g., to update dependent params
	autosave                       // allow it to be saved
}

func param(nComp int, name, unit string) Param {
	var p Param
	p.autosave = newAutosave(nComp, name, unit, Mesh())
	p.gpu = make(cuda.LUTPtrs, nComp)
	for i := range p.gpu {
		p.gpu[i] = cuda.MemAlloc(LUTSIZE * cu.SIZEOF_FLOAT32)
	}
	p.lut = make([][LUTSIZE]float32, nComp)
	p.ok = false
	return p
}

func (p *Param) SetRegion(region int, v ...float32) {
	util.Argument(len(v) == p.NComp())
	if region == 0 {
		log.Fatal("cannot set parameters in region 0 (vacuum)")
	}
	for c := range v {
		p.lut[c][region] = v[c]
	}
	p.ok = false
	if p.post_update != nil {
		p.post_update(region)
	}
}

func (p *Param) GetRegion(region int) []float32 {
	v := make([]float32, p.NComp())
	for c := range v {
		v[c] = p.lut[c][region]
	}
	return v
}

// Get a GPU mirror of the look-up table.
// Copies to GPU first only if needed.
func (p *Param) Gpu() cuda.LUTPtrs {
	if p.ok {
		return p.gpu
	}
	p.upload()
	return p.gpu
}

// XYZ swap here
func (p *Param) upload() {
	log.Println("upload LUT", p.name, p.lut)
	for c2 := range p.gpu {
		c := util.SwapIndex(c2, p.NComp())
		cu.MemcpyHtoD(cu.DevicePtr(p.gpu[c]), unsafe.Pointer(&p.lut[c2][0]), cu.SIZEOF_FLOAT32*LUTSIZE)
	}
	p.ok = true
}

// scale vector by a, overwrite source and return it for convenience
func scale(v []float32, a float32) []float32 {
	for i := range v {
		v[i] *= a
	}
	return v
}

// rudimentary api
func RegionSetVector(region int, p *Param, v [3]float64) {
	p.SetRegion(region, []float32{float32(v[0]), float32(v[1]), float32(v[2])}...)
}

func init() {
	world.Func("RegionSetVector", RegionSetVector)
}
