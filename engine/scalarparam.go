package engine

import (
	"github.com/mumax/3/script"
	"reflect"
)

// specialized param with 1 component
type ScalarParam struct {
	inputParam
}

func (p *ScalarParam) init(name, unit, desc string, children []derived) {
	p.inputParam.init(SCALAR, name, unit, children)
	DeclLValue(name, p, cat(desc, unit))
}

func (p *ScalarParam) SetRegion(region int, f script.ScalarFunction) {
	if region == -1 {
		p.setRegionsFunc(0, NREGION, f) // uniform
	} else {
		p.setRegionsFunc(region, region+1, f) // upper bound exclusive
	}
}

func (p *ScalarParam) SetValue(v interface{}) {
	f := v.(script.ScalarFunction)
	p.setRegionsFunc(0, NREGION, f)
}

func (p *ScalarParam) Set(v float64) {
	p.setRegions(0, NREGION, []float64{v})
}

func (p *ScalarParam) setRegionsFunc(r1, r2 int, f script.ScalarFunction) {
	if Const(f) {
		p.setRegions(r1, r2, []float64{f.Float()})
	} else {
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
func (p *ScalarParam) Average() float64        { return qAverageUniverse(p)[0] }
func (p *ScalarParam) Region(r int) *sOneReg   { return sOneRegion(p, r) }
func (p *ScalarParam) Child() []script.Expr    { return nil }

// checks if a script expression contains t (time)
func Const(e script.Expr) bool {
	t := World.Resolve("t")
	return !script.Contains(e, t)
}

func cat(desc, unit string) string {
	if unit == "" {
		return desc
	} else {
		return desc + " (" + unit + ")"
	}
}
