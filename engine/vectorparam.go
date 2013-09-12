package engine

import "reflect"

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

func (p *VectorParam) SetValue(v interface{}) {
	vec := v.([3]float64)
	p.setUniform(vec[:])
}

func (p *VectorParam) Eval() interface{}       { return p }
func (p *VectorParam) Type() reflect.Type      { return reflect.TypeOf(new(VectorParam)) }
func (p *VectorParam) InputType() reflect.Type { return reflect.TypeOf([3]float64{}) }
