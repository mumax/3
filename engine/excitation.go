package engine

import (
	"math"
	"reflect"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/script"
	"github.com/mumax/3/util"
)

// space-dependent mask plus time dependent multiplier
type mulmask struct {
	mul  func() float64
	mask *data.Slice
}

type Excitation interface {
	guaranteedTimeIndependent() bool
}

// An excitation, typically field or current,
// can be defined region-wise plus extra mask*multiplier terms.
type VectorExcitation struct {
	name       string
	perRegion  RegionwiseVector // Region-based excitation
	extraTerms []mulmask        // add extra mask*multiplier terms
}

func NewVectorExcitation(name, unit, desc string) *VectorExcitation {
	e := new(VectorExcitation)
	e.name = name
	e.perRegion.init(3, "_"+name+"_perRegion", unit, nil) // name starts with underscore: unexported
	DeclLValue(name, e, cat(desc, unit))
	return e
}

func (p *VectorExcitation) MSlice() cuda.MSlice {
	buf, r := p.Slice()
	util.Assert(r == true)
	return cuda.ToMSlice(buf)
}

func (e *VectorExcitation) AddTo(dst *data.Slice) {
	if !e.perRegion.isZero() {
		cuda.RegionAddV(dst, e.perRegion.gpuLUT(), regions.Gpu())
	}

	for _, t := range e.extraTerms {
		var mul float32 = 1
		if t.mul != nil {
			mul = float32(t.mul())
		}
		cuda.Madd2(dst, dst, t.mask, 1, mul)
	}
}

func (e *VectorExcitation) isZero() bool {
	return e.perRegion.isZero() && len(e.extraTerms) == 0
}

func (e *VectorExcitation) Slice() (*data.Slice, bool) {
	buf := cuda.Buffer(e.NComp(), e.Mesh().Size())
	cuda.Zero(buf)
	e.AddTo(buf)
	return buf, true
}

// After resizing the mesh, the extra terms don't fit the grid anymore
// and there is no reasonable way to resize them. So remove them and have
// the user re-add them.
func (e *VectorExcitation) RemoveExtraTerms() {
	if len(e.extraTerms) == 0 {
		return
	}

	LogOut("REMOVING EXTRA TERMS FROM", e.Name())
	for _, m := range e.extraTerms {
		m.mask.Free()
	}
	e.extraTerms = nil
}

// Returns true if the excitation e is guaranteed to be constant in time in all regions.
// Conversely, returning false does not necessarily mean the excitation is time-dependent.
func (e *VectorExcitation) guaranteedTimeIndependent() bool {
	if len(e.extraTerms) > 0 {
		return false
	}
	for r := range NREGION {
		if e.perRegion.upd_reg[r] != nil {
			return false
		}
	}
	return true
}

// Add an extra mask*multiplier term to the excitation.
func (e *VectorExcitation) Add(mask *data.Slice, f script.ScalarFunction) {
	var mul func() float64
	if f != nil {
		if IsConst(f) {
			val := f.Float()
			mul = func() float64 {
				return val
			}
		} else {
			mul = func() float64 {
				return f.Float()
			}
		}
	}
	e.AddGo(mask, mul)
}

// An Add(mask, f) equivalent for Go use
func (e *VectorExcitation) AddGo(mask *data.Slice, mul func() float64) {
	if mask != nil {
		checkNaN(mask, e.Name()+".add()") // TODO: in more places
		mask = data.Resample(mask, e.Mesh().Size())
		mask = assureGPU(mask)
	}
	e.extraTerms = append(e.extraTerms, mulmask{mul, mask})
}

func (e *VectorExcitation) SetRegion(region int, f script.VectorFunction) {
	e.perRegion.SetRegion(region, f)
}
func (e *VectorExcitation) SetValue(v interface{})         { e.perRegion.SetValue(v) }
func (e *VectorExcitation) Set(v data.Vector)              { e.perRegion.setRegions(0, NREGION, slice(v)) }
func (e *VectorExcitation) getRegion(region int) []float64 { return e.perRegion.getRegion(region) } // for gui

func (e *VectorExcitation) SetRegionFn(region int, f func() [3]float64) {
	e.perRegion.setFunc(region, region+1, func() []float64 {
		return slice(f())
	})
}

func (e *VectorExcitation) average() []float64      { return qAverageUniverse(e) }
func (e *VectorExcitation) Average() data.Vector    { return unslice(qAverageUniverse(e)) }
func (e *VectorExcitation) IsUniform() bool         { return e.perRegion.IsUniform() }
func (e *VectorExcitation) Name() string            { return e.name }
func (e *VectorExcitation) Unit() string            { return e.perRegion.Unit() }
func (e *VectorExcitation) NComp() int              { return e.perRegion.NComp() }
func (e *VectorExcitation) Mesh() *data.Mesh        { return Mesh() }
func (e *VectorExcitation) Region(r int) *vOneReg   { return vOneRegion(e, r) }
func (e *VectorExcitation) Comp(c int) ScalarField  { return Comp(e, c) }
func (e *VectorExcitation) Eval() interface{}       { return e }
func (e *VectorExcitation) Type() reflect.Type      { return reflect.TypeOf(new(VectorExcitation)) }
func (e *VectorExcitation) InputType() reflect.Type { return script.VectorFunction_t }
func (e *VectorExcitation) EvalTo(dst *data.Slice)  { EvalTo(e, dst) }

// An excitation, typically field or current,
// can be defined region-wise plus extra mask*multiplier terms.
type ScalarExcitation struct {
	name       string
	perRegion  RegionwiseScalar // Region-based excitation
	extraTerms []mulmask        // add extra mask*multiplier terms
}

func NewScalarExcitation(name, unit, desc string) *ScalarExcitation {
	e := new(ScalarExcitation)
	e.name = name
	e.perRegion.init("_"+name+"_perRegion", unit, desc, nil) // name starts with underscore: unexported
	DeclLValue(name, e, cat(desc, unit))
	return e
}

func (p *ScalarExcitation) MSlice() cuda.MSlice {
	buf, r := p.Slice()
	util.Assert(r == true)
	return cuda.ToMSlice(buf)
}

func (e *ScalarExcitation) AddTo(dst *data.Slice) {
	if !e.perRegion.isZero() {
		cuda.RegionAddS(dst, e.perRegion.gpuLUT1(), regions.Gpu())
	}

	for _, t := range e.extraTerms {
		var mul float32 = 1
		if t.mul != nil {
			mul = float32(t.mul())
		}
		cuda.Madd2(dst, dst, t.mask, 1, mul)
	}
}

func (e *ScalarExcitation) isZero() bool {
	return e.perRegion.isZero() && len(e.extraTerms) == 0
}

func (e *ScalarExcitation) Slice() (*data.Slice, bool) {
	buf := cuda.Buffer(e.NComp(), e.Mesh().Size())
	cuda.Zero(buf)
	e.AddTo(buf)
	return buf, true
}

// After resizing the mesh, the extra terms don't fit the grid anymore
// and there is no reasonable way to resize them. So remove them and have
// the user re-add them.
func (e *ScalarExcitation) RemoveExtraTerms() {
	if len(e.extraTerms) == 0 {
		return
	}

	LogOut("REMOVING EXTRA TERMS FROM", e.Name())
	for _, m := range e.extraTerms {
		m.mask.Free()
	}
	e.extraTerms = nil
}

// Returns true if the excitation e is guaranteed to be constant in time in all regions.
// Conversely, returning false does not necessarily mean the excitation is time-dependent.
func (e *ScalarExcitation) guaranteedTimeIndependent() bool {
	if len(e.extraTerms) > 0 {
		return false
	}
	for r := range NREGION {
		if e.perRegion.upd_reg[r] != nil {
			return false
		}
	}
	return true
}

// Add an extra mask*multiplier term to the excitation.
func (e *ScalarExcitation) Add(mask *data.Slice, f script.ScalarFunction) {
	var mul func() float64
	if f != nil {
		if IsConst(f) {
			val := f.Float()
			mul = func() float64 {
				return val
			}
		} else {
			mul = func() float64 {
				return f.Float()
			}
		}
	}
	e.AddGo(mask, mul)
}

// An Add(mask, f) equivalent for Go use
func (e *ScalarExcitation) AddGo(mask *data.Slice, mul func() float64) {
	if mask != nil {
		checkNaN(mask, e.Name()+".add()") // TODO: in more places
		mask = data.Resample(mask, e.Mesh().Size())
		mask = assureGPU(mask)
	}
	e.extraTerms = append(e.extraTerms, mulmask{mul, mask})
}

func (e *ScalarExcitation) SetRegion(region int, f script.ScalarFunction) {
	e.perRegion.SetRegion(region, f)
}
func (e *ScalarExcitation) SetValue(v interface{})         { e.perRegion.SetValue(v) }
func (e *ScalarExcitation) Set(v float64)                  { e.perRegion.setRegions(0, NREGION, []float64{v}) }
func (e *ScalarExcitation) getRegion(region int) []float64 { return e.perRegion.getRegion(region) } // for gui

func (e *ScalarExcitation) SetRegionFn(region int, f func() [3]float64) {
	e.perRegion.setFunc(region, region+1, func() []float64 {
		return slice(f())
	})
}

func (e *ScalarExcitation) average() float64        { return qAverageUniverse(e)[0] }
func (e *ScalarExcitation) Average() float64        { return e.average() }
func (e *ScalarExcitation) IsUniform() bool         { return e.perRegion.IsUniform() }
func (e *ScalarExcitation) Name() string            { return e.name }
func (e *ScalarExcitation) Unit() string            { return e.perRegion.Unit() }
func (e *ScalarExcitation) NComp() int              { return e.perRegion.NComp() }
func (e *ScalarExcitation) Mesh() *data.Mesh        { return Mesh() }
func (e *ScalarExcitation) Region(r int) *vOneReg   { return vOneRegion(e, r) }
func (e *ScalarExcitation) Comp(c int) ScalarField  { return Comp(e, c) }
func (e *ScalarExcitation) Eval() interface{}       { return e }
func (e *ScalarExcitation) Type() reflect.Type      { return reflect.TypeOf(new(ScalarExcitation)) }
func (e *ScalarExcitation) InputType() reflect.Type { return script.ScalarFunction_t }
func (e *ScalarExcitation) EvalTo(dst *data.Slice)  { EvalTo(e, dst) }

func checkNaN(s *data.Slice, name string) {
	h := s.Host()
	for _, h := range h {
		for _, v := range h {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				util.Fatalf("NaN or Inf in %s", name)
			}
		}
	}
}
