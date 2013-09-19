package engine

import (
	"log"
	"reflect"
)

// specialized param with 1 component
type ScalarParam struct {
	inputParam
}

func (p *ScalarParam) init(name, unit, desc string, children []derived) {
	p.inputParam.init(1, name, unit, children)
	DeclLValue(name, p, desc)
}

func (p *ScalarParam) SetRegion(region int, value float64) {
	p.setRegion(region, []float64{value})
}

func (p *ScalarParam) GetRegion(region int) float64 {
	return float64(p.getRegion(region)[0])
}

func (p *ScalarParam) SetValue(v interface{}) {
	log.Println(p.Name(), ".SetValue", v)
	p.setUniform([]float64{v.(Func).Float()})
}

type Func interface {
	Float() float64
}

func (p *ScalarParam) Eval() interface{}       { return p }
func (p *ScalarParam) Type() reflect.Type      { return reflect.TypeOf(new(ScalarParam)) }
func (p *ScalarParam) InputType() reflect.Type { return func_t }

// maneuvers to get interface type of Func (simpler way?)
var func_t = reflect.TypeOf(dummy_f).In(0)

func dummy_f(Func) {}
