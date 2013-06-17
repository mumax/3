package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
	"log"
	"reflect"
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
	util.Argument(len(v) == p.NComp()) // note: also likely panics if param not initialized (nComp = 0)

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

// check if region is OK for use.
func checkRegion(region int) {
	if region == 0 {
		log.Fatal("cannot set parameters in region 0 (vacuum)")
	}
	if !regions.defined[region] {
		log.Panic("region ", region, " has not yet been defined by DefRegion()")
	}
}

// Get returns the space-dependent parameter as a slice of floats, so it can be output.
func (p *param) Get() (*data.Slice, bool) {
	s := data.NewSlice(p.NComp(), p.Mesh())
	l := s.Host()

	for i, r := range regions.cpu {
		v := p.getRegion(int(r))
		for c := range l {
			l[c][i] = float32(v[c])
		}
	}
	return s, false
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
	log.Println("upload LUT", p.name, p.lut)
	for c2 := range p.gpu {
		c := util.SwapIndex(c2, p.NComp())
		cu.MemcpyHtoD(cu.DevicePtr(p.gpu[c]), unsafe.Pointer(&p.lut[c2][0]), cu.SIZEOF_FLOAT32*MAXREG)
	}
	p.ok = true
}

type ScalarParam struct{ param }

func scalarParam(name, unit string, post func(int)) ScalarParam {
	return ScalarParam{newParam(1, name, unit, post)}
}
func (p *ScalarParam) SetRegion(region int, value float64) {
	checkRegion(region)
	p.setRegion(region, value)
}
func (p *ScalarParam) SetValue(v interface{}) {
	p.setUniform(v.(float64))
}
func (p *ScalarParam) Eval() interface{}            { return p }
func (p *ScalarParam) Type() reflect.Type           { return reflect.TypeOf(new(ScalarParam)) }
func (p *ScalarParam) InputType() reflect.Type      { return reflect.TypeOf(float64(0)) }
func (p *ScalarParam) GetRegion(region int) float64 { return float64(p.lut[0][region]) }
func (p *ScalarParam) GetUniform() float64          { return p.getUniform()[0] }
func (p *ScalarParam) Gpu() cuda.LUTPtr             { return cuda.LUTPtr(p.param.Gpu()[0]) }
func (p *ScalarParam) Set(v float64)                { p.setUniform(v) }

type VectorParam struct{ param }

func vectorParam(name, unit string, post func(int)) VectorParam {
	return VectorParam{newParam(3, name, unit, post)}
}

func (p *VectorParam) SetRegion(region int, value [3]float64) {
	checkRegion(region)
	p.setRegion(region, value[:]...)
}
func (p *VectorParam) SetValue(v interface{}) {
	vec := v.([3]float64)
	p.setUniform(vec[:]...)
}
func (p *VectorParam) Eval() interface{}       { return p }
func (p *VectorParam) Type() reflect.Type      { return reflect.TypeOf(new(VectorParam)) }
func (p *VectorParam) InputType() reflect.Type { return reflect.TypeOf([3]float64{}) }
