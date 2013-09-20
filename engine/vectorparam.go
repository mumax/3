package engine

import (
	"code.google.com/p/mx3/script"
	"reflect"
)

type VectorParam struct {
	inputParam
}

func (p *VectorParam) init(name, unit, desc string) {
	p.inputParam.init(3, name, unit, nil) // no vec param has children (yet)
	DeclLValue(name, p, desc)
}

func (p *VectorParam) SetRegion(region int, value [3]float64) {
	//checkRegion(region)
	p.setRegion(region, value[:])
}

func (p *VectorParam) GetRegion(region int) [3]float64 {
	v := p.getRegion(region)
	return [3]float64{v[0], v[1], v[2]}
}

func (p *VectorParam) SetValue(v interface{}) {
	f := v.(script.VectorFunction)
	if f.Const() {
		p.setUniform(slice(f.Float3()))
	} else {
		p.setFunc(0, NREGION, func() []float64 {
			return slice(f.Float3())
		})
	}
}

func (p *VectorParam) Eval() interface{}       { return p }
func (p *VectorParam) Type() reflect.Type      { return reflect.TypeOf(new(VectorParam)) }
func (p *VectorParam) InputType() reflect.Type { return script.VectorFunction_t }

// shortcut for slicing unaddressable_vector()[:]
func slice(v [3]float64) []float64 {
	return v[:]
}
