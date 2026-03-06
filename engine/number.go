package engine

import (
	"reflect"
)

// TODO: wrap around outputValue

// inputValue is like outputValue, but settable
type inputValue struct {
	v          float64
	onSet      func()
	name, unit string
}

func numParam(v float64, name, unit string, onSet func()) inputValue {
	return inputValue{v: v, onSet: onSet, name: name, unit: unit}
}

func (p *inputValue) NComp() int              { return 1 }
func (p *inputValue) Name() string            { return p.name }
func (p *inputValue) Unit() string            { return p.unit }
func (p *inputValue) getRegion(int) []float64 { return []float64{float64(p.v)} }
func (p *inputValue) Type() reflect.Type      { return reflect.TypeFor[float64]() }
func (p *inputValue) IsUniform() bool         { return true }
func (p *inputValue) Eval() any               { return p.v }

func (p *inputValue) SetValue(v any) {
	p.v = v.(float64)
	p.onSet()
}
