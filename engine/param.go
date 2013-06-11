package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
	"log"
	"reflect"
	"unsafe"
)

// param stores a space-dependent material parameter
// by keeping a look-up table mapping region index to float value.
type param struct {
	lut         [][MAXREG]float32 // look-up table source
	gpu         cuda.LUTPtrs      // gpu copy of lut, lazily transferred when needed
	ok          bool              // gpu cache up-to date with lut source
	zero        bool              // are all values zero (then we may skip corresponding kernel)
	post_update func(region int)  // called after region value changed, e.g., to update dependent params
	autosave                      // allow it to be saved
}

// constructor
func newParam(nComp int, name, unit string) param {
	var p param
	p.autosave = newAutosave(nComp, name, unit, Mesh())
	p.gpu = make(cuda.LUTPtrs, nComp)
	for i := range p.gpu {
		p.gpu[i] = cuda.MemAlloc(MAXREG * cu.SIZEOF_FLOAT32)
	}
	p.lut = make([][MAXREG]float32, nComp)
	p.ok = false
	return p
}

func (p *param) setRegion(region int, v ...float64) {
	util.Argument(len(v) == p.NComp())
	if region == 0 {
		log.Fatal("cannot set parameters in region 0 (vacuum)")
	}
	if !regions.defined[region] {
		log.Fatal("region ", region, " has not yet been defined by DefRegion()")
	}
	for c := range v {
		p.lut[c][region] = float32(v[c])
	}
	p.ok = false
	if p.post_update != nil {
		p.post_update(region)
	}
}

func (p *param) GetRegion(region int) []float64 {
	v := make([]float64, p.NComp())
	for c := range v {
		v[c] = float64(p.lut[c][region])
	}
	return v
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
	log.Println("upload LUT", p.name, p.lut)
	for c2 := range p.gpu {
		c := util.SwapIndex(c2, p.NComp())
		cu.MemcpyHtoD(cu.DevicePtr(p.gpu[c]), unsafe.Pointer(&p.lut[c2][0]), cu.SIZEOF_FLOAT32*MAXREG)
	}
	p.ok = true
}

// scale vector by a, overwrite source and return it for convenience
func scale(v []float64, a float64) []float64 {
	for i := range v {
		v[i] *= a
	}
	return v
}

type ScalarParam struct{ param }

func (s *ScalarParam) SetRegion(region int, value float64) { s.param.setRegion(region, value) }
func (p *ScalarParam) SetValue(v interface{})              { p.SetRegion(1, v.(float64)) }
func (p *ScalarParam) Eval() interface{}                   { return p }
func (p *ScalarParam) Type() reflect.Type                  { return reflect.TypeOf(new(ScalarParam)) }
func (p *ScalarParam) InputType() reflect.Type             { return reflect.TypeOf(float64(0)) }

type VectorParam struct{ param }

func vectorParam(name, unit string) VectorParam {
	return VectorParam{newParam(3, name, unit)}
}
func (s *VectorParam) SetRegion(region int, value [3]float64) { s.setRegion(region, value[:]...) }
func (p *VectorParam) SetValue(v interface{})                 { vec := v.([3]float64); p.setRegion(1, vec[:]...) }
func (p *VectorParam) Eval() interface{}                      { return p }
func (p *VectorParam) Type() reflect.Type                     { return reflect.TypeOf(new(VectorParam)) }
func (p *VectorParam) InputType() reflect.Type                { return reflect.TypeOf([3]float64{}) }
