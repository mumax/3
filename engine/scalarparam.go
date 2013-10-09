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
	DeclLValue(name, p, cat(desc, unit))
}

func cat(desc, unit string) string {
	if unit == "" {
		return desc
	} else {
		return desc + " (" + unit + ")"
	}
}

func (p *ScalarParam) SetRegion(region int, f script.ScalarFunction) {
	p.setRegionsFunc(region, region+1, f)
}

func (p *ScalarParam) SetValue(v interface{}) {
	f := v.(script.ScalarFunction)
	p.setRegionsFunc(0, NREGION, f)
}

func (p *ScalarParam) setRegionsFunc(r1, r2 int, f script.ScalarFunction) {
	if f.Const() {
		log.Println(p.Name(), "[", r1, ":", r2, "]", "is constant")
		p.setRegions(r1, r2, []float64{f.Float()})
	} else {
		log.Println(p.Name(), "[", r1, ":", r2, "]", "is not constant")
		p.setFunc(r1, r2, func() []float64 {
			return []float64{f.Float()}
		})
	}
}

func (p *ScalarParam) GetRegion(region int) float64 {
	return float64(p.getRegion(region)[0])
}

func (p *ScalarParam) Eval() interface{}       { return p }
func (p *ScalarParam) Type() reflect.Type      { return reflect.TypeOf(new(ScalarParam)) }
func (p *ScalarParam) InputType() reflect.Type { return script.ScalarFunction_t }
