package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"reflect"
)

// An excitation, typically field or current,
// can be defined region-wise plus extra mask*multiplier terms.
type excitation struct {
	perRegion  inputParam // Region-based excitation
	extraTerms []mulmask  // add extra mask*multiplier terms
}

type mulmask struct {
	mul  func() float64
	mask *data.Slice
}

func (e *excitation) init(name, unit, desc string) {
	e.perRegion.init(3, name, unit, nil)
	DeclLValue(name, e, desc)
}

func (e *excitation) addTo(dst *data.Slice) {
	if !e.perRegion.isZero() {
		cuda.RegionAddV(dst, e.perRegion.LUT(), regions.Gpu())
	}
	for _, t := range e.extraTerms {
		cuda.Madd2(dst, dst, t.mask, 1, float32(t.mul()))
	}
}

func (e *excitation) isZero() bool {
	return e.perRegion.isZero() && len(e.extraTerms) == 0
}

func (e *excitation) Get() (*data.Slice, bool) {
	buf := cuda.Buffer(e.NComp(), e.Mesh())
	cuda.Zero(buf)
	e.addTo(buf)
	return buf, true
}

// Add an extra maks*multiplier term to the excitation.
func (e *excitation) Add(mask *data.Slice, mul func() float64) {
	e.extraTerms = append(e.extraTerms, mulmask{mul, assureGPU(mask)})
}

func assureGPU(s *data.Slice) *data.Slice {
	if s.GPUAccess() {
		return s
	} else {
		return cuda.GPUCopy(s)
	}
}

// user script: has to be 3-vector
func (e *excitation) SetRegion(region int, value [3]float64) {
	e.perRegion.setRegion(region, value[:])
}

// for gui (nComp agnostic)
func (e *excitation) setRegion(region int, value []float64) {
	e.perRegion.setRegion(region, value)
}

// does not use extramask!
func (e *excitation) getRegion(region int) []float64 {
	return e.perRegion.getRegion(region)
}

//func (e *excitation) GetVec() []float64 {
//	if len(e.extraTerms) != 0 {
//		panic(e.Name() + " is space-dependent, cannot be used as value")
//	}
//	return e.perRegion.getRegion(0)
//}

// needed for script
func (e *excitation) SetValue(v interface{}) {
	vec := v.([3]float64)
	e.perRegion.setUniform(vec[:])
}

func (e *excitation) Name() string            { return e.perRegion.Name() }
func (e *excitation) Unit() string            { return e.perRegion.Unit() }
func (e *excitation) NComp() int              { return e.perRegion.NComp() }
func (e *excitation) Mesh() *data.Mesh        { return &globalmesh }
func (e *excitation) Eval() interface{}       { return e }
func (e *excitation) Type() reflect.Type      { return reflect.TypeOf(new(excitation)) }
func (e *excitation) InputType() reflect.Type { return reflect.TypeOf([3]float64{}) }
