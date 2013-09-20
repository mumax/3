package engine

import (
	"github.com/mumax/3/script"
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
	f := v.(script.ScalarFunction)
	if f.Const() {
		p.setUniform([]float64{f.Float()})
	} else {
		p.setFunc(0, NREGION, func() []float64 {
			return []float64{f.Float()}
		})
	}
}

func (p *ScalarParam) Eval() interface{}       { return p }
func (p *ScalarParam) Type() reflect.Type      { return reflect.TypeOf(new(ScalarParam)) }
func (p *ScalarParam) InputType() reflect.Type { return script.ScalarFunction_t }
