package engine

import (
	"github.com/mumax/3/v3/cuda"
	"github.com/mumax/3/v3/data"
	"github.com/mumax/3/v3/script"
	"github.com/mumax/3/v3/util"
	"math"
	"reflect"
)

// An excitation, typically field or current,
// can be defined region-wise plus extra mask*multiplier terms.
type Excitation struct {
	name       string
	perRegion  RegionwiseVector // Region-based excitation
	extraTerms []mulmask        // add extra mask*multiplier terms
}

// space-dependent mask plus time dependent multiplier
type mulmask struct {
	mul  func() float64
	mask *data.Slice
}

func NewExcitation(name, unit, desc string) *Excitation {
	e := new(Excitation)
	e.name = name
	e.perRegion.init(3, "_"+name+"_perRegion", unit, nil) // name starts with underscore: unexported
	DeclLValue(name, e, cat(desc, unit))
	return e
}

func (p *Excitation) MSlice() cuda.MSlice {
	buf, r := p.Slice()
	util.Assert(r == true)
	return cuda.ToMSlice(buf)
}

func (e *Excitation) AddTo(dst *data.Slice) {
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

func (e *Excitation) isZero() bool {
	return e.perRegion.isZero() && len(e.extraTerms) == 0
}

func (e *Excitation) Slice() (*data.Slice, bool) {
	buf := cuda.Buffer(e.NComp(), e.Mesh().Size())
	cuda.Zero(buf)
	e.AddTo(buf)
	return buf, true
}

// After resizing the mesh, the extra terms don't fit the grid anymore
// and there is no reasonable way to resize them. So remove them and have
// the user re-add them.
func (e *Excitation) RemoveExtraTerms() {
	if len(e.extraTerms) == 0 {
		return
	}

	LogOut("REMOVING EXTRA TERMS FROM", e.Name())
	for _, m := range e.extraTerms {
		m.mask.Free()
	}
	e.extraTerms = nil
}

// Add an extra mask*multiplier term to the excitation.
func (e *Excitation) Add(mask *data.Slice, f script.ScalarFunction) {
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
func (e *Excitation) AddGo(mask *data.Slice, mul func() float64) {
	if mask != nil {
		checkNaN(mask, e.Name()+".add()") // TODO: in more places
		mask = data.Resample(mask, e.Mesh().Size())
		mask = assureGPU(mask)
	}
	e.extraTerms = append(e.extraTerms, mulmask{mul, mask})
}

func (e *Excitation) SetRegion(region int, f script.VectorFunction) { e.perRegion.SetRegion(region, f) }
func (e *Excitation) SetValue(v interface{})                        { e.perRegion.SetValue(v) }
func (e *Excitation) Set(v data.Vector)                             { e.perRegion.setRegions(0, NREGION, slice(v)) }
func (e *Excitation) getRegion(region int) []float64                { return e.perRegion.getRegion(region) } // for gui

func (e *Excitation) SetRegionFn(region int, f func() [3]float64) {
	e.perRegion.setFunc(region, region+1, func() []float64 {
		return slice(f())
	})
}

func (e *Excitation) average() []float64      { return qAverageUniverse(e) }
func (e *Excitation) Average() data.Vector    { return unslice(qAverageUniverse(e)) }
func (e *Excitation) IsUniform() bool         { return e.perRegion.IsUniform() }
func (e *Excitation) Name() string            { return e.name }
func (e *Excitation) Unit() string            { return e.perRegion.Unit() }
func (e *Excitation) NComp() int              { return e.perRegion.NComp() }
func (e *Excitation) Mesh() *data.Mesh        { return Mesh() }
func (e *Excitation) Region(r int) *vOneReg   { return vOneRegion(e, r) }
func (e *Excitation) Comp(c int) ScalarField  { return Comp(e, c) }
func (e *Excitation) Eval() interface{}       { return e }
func (e *Excitation) Type() reflect.Type      { return reflect.TypeOf(new(Excitation)) }
func (e *Excitation) InputType() reflect.Type { return script.VectorFunction_t }
func (e *Excitation) EvalTo(dst *data.Slice)  { EvalTo(e, dst) }

func checkNaN(s *data.Slice, name string) {
	h := s.Host()
	for _, h := range h {
		for _, v := range h {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				util.Fatal("NaN or Inf in", name)
			}
		}
	}
}
