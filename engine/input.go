package engine

/*
parameters are region- and time dependent input values,
like material parameters.
*/

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/script"
	"github.com/mumax/3/util"
	"reflect"
)

type input struct {
	q          Q
	name, unit string
}

type ScalarInput struct {
	input
}

type VectorInput struct {
	input
}

func (p *input) SetQ(q Q) {
	util.Argument(p.q.NComp() == q.NComp())
	p.q = q
}

func (p *input) EvalTo(dst *data.Slice)     { p.q.EvalTo(dst) }
func (p *input) Slice() (*data.Slice, bool) { return ValueOf(p), true }
func (p *input) Name() string               { return p.name }
func (p *input) Unit() string               { return p.unit }
func (p *input) Mesh() *data.Mesh           { return Mesh() }
func (p *input) NComp() int                 { return p.q.NComp() }

func (p *input) IsUniform() bool {
	return false // TODO
}

func (p *regionwise) average() []float64 { return qAverageUniverse(p) }

func NewScalarInput(name, unit, desc string) *ScalarInput {
	s := new(RegionwiseScalar)
	s.regionwise.init(SCALAR, name, unit, nil)
	p := &ScalarInput{input{Const(0), name, unit}}
	DeclLValue(name, p, cat(desc, unit))
	return p
}

func NewVectorInput(name, unit, desc string) *VectorInput {
	s := new(RegionwiseVector)
	s.regionwise.init(VECTOR, name, unit, nil)
	p := &VectorInput{input{ConstVector(0, 0, 0), name, unit}}
	DeclLValue(name, p, cat(desc, unit))
	return p
}

func (p *ScalarInput) Set(v float64) {
	// todo
}

func (p *ScalarInput) Eval() interface{}       { return p }
func (p *ScalarInput) Type() reflect.Type      { return reflect.TypeOf(new(ScalarInput)) }
func (p *ScalarInput) InputType() reflect.Type { return script.ScalarFunction_t }
func (p *ScalarInput) Average() float64        { return AverageOf(p.q)[0] }

func (p *ScalarInput) SetValue(v interface{}) {
	f := v.(script.ScalarFunction)
	if IsConst(f) {
		p.SetQ(Const(f.Float()))
	} else {
		f := f.Fix() // fix values of all variables except t
		p.SetQ(Func(func() float64 {
			return f.Eval().(script.ScalarFunction).Float()
		}))
	}
}

type Func func() float64

func (f Func) EvalTo(dst *data.Slice) {
	util.Argument(dst.NComp() == f.NComp())
	cuda.Memset(dst, float32(f()))
}

func (f Func) NComp() int {
	return 1
}

var _ Q = Func(nil)

type VectorFunc func() data.Vector

func (f VectorFunc) EvalTo(dst *data.Slice) {
	util.Argument(dst.NComp() == f.NComp())
	v := f()
	for i, v := range v {
		cuda.Memset(dst.Comp(i), float32(v))
	}
}

func (f VectorFunc) NComp() int {
	return 3
}

var _ Q = VectorFunc(nil)

func (p *VectorInput) Eval() interface{}       { return p }
func (p *VectorInput) Type() reflect.Type      { return reflect.TypeOf(new(VectorInput)) }
func (p *VectorInput) InputType() reflect.Type { return script.VectorFunction_t }
func (p *VectorInput) Region(r int) *vOneReg   { return vOneRegion(p, r) }
func (p *VectorInput) Average() data.Vector    { return unslice(qAverageUniverse(p)) }
func (p *VectorInput) Comp(c int) ScalarField  { return Comp(p, c) }
func (p *VectorInput) average() []float64      { return AverageOf(p.q) }

func (p *VectorInput) SetValue(v interface{}) {
	f := v.(script.VectorFunction)
	if IsConst(f) {
		v := f.Float3()
		p.SetQ(ConstVector(v.X(), v.Y(), v.Z()))
	} else {
		f := f.Fix() // fix values of all variables except t
		p.SetQ(VectorFunc(func() data.Vector {
			return f.Eval().(script.VectorFunction).Float3()
		}))
	}
}
