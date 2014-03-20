package engine

import (
	"reflect"
)

type numberParam struct {
	v          float64
	onSet      func()
	name, unit string
}

func numParam(v float64, name, unit string, onSet func()) numberParam {
	return numberParam{v: v, onSet: onSet, name: name, unit: unit}
}

func (p *numberParam) NComp() int              { return 1 }
func (p *numberParam) Name() string            { return p.name }
func (p *numberParam) Unit() string            { return p.unit }
func (p *numberParam) getRegion(int) []float64 { return []float64{float64(p.v)} }
func (p *numberParam) Type() reflect.Type      { return reflect.TypeOf(float64(0)) }
func (p *numberParam) IsUniform() bool         { return true }
func (p *numberParam) Eval() interface{}       { return p.v }

func (p *numberParam) SetValue(v interface{}) {
	p.v = v.(float64)
	p.onSet()
}
