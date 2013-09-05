package engine

import (
	"code.google.com/p/mx3/cuda"
	"reflect"
)

// specialized param with 1 component
type ScalarParam struct {
	param
}

func scalarParam(name, unit string, deps ...*param) ScalarParam {
	return ScalarParam{newParam(1, name, unit, deps...)}
}

func (p *ScalarParam) SetRegion(region int, value float64) {
	//checkRegion(region)
	p.setRegion(region, value)
}

func (p *ScalarParam) SetValue(v interface{}) {
	p.setUniform(v.(float64))
}

func (p *ScalarParam) Eval() interface{}            { return p }
func (p *ScalarParam) Type() reflect.Type           { return reflect.TypeOf(new(ScalarParam)) }
func (p *ScalarParam) InputType() reflect.Type      { return reflect.TypeOf(float64(0)) }
func (p *ScalarParam) GetRegion(region int) float64 { return float64(p.cpu[0][region]) }
func (p *ScalarParam) GetUniform() float64          { return p.getUniform()[0] }
func (p *ScalarParam) Gpu() cuda.LUTPtr             { return cuda.LUTPtr(p.param.Gpu()[0]) }
func (p *ScalarParam) Set(v float64)                { p.setUniform(v) }
func (p *ScalarParam) GetFloat() float64            { return p.GetUniform() }

func (p *ScalarParam) SetFunc(f func() float64) {
	// TODO: time - dependent
	panic("todo")
}
