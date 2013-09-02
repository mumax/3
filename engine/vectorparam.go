package engine

import (
	"reflect"
)

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
