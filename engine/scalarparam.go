package engine

import (
	"github.com/mumax/3/script"
	"reflect"
)

// specialized param with 1 component
type ScalarInput struct {
	inputParam
}

func (p *ScalarInput) init(name, unit, desc string, children []derived) {
	p.inputParam.init(SCALAR, name, unit, children)
	DeclLValue(name, p, cat(desc, unit))
}

func (p *ScalarInput) SetRegion(region int, f script.ScalarFunction) {
	if region == -1 {
		p.setRegionsFunc(0, NREGION, f) // uniform
	} else {
		p.setRegionsFunc(region, region+1, f) // upper bound exclusive
	}
}

func (p *ScalarInput) SetValue(v interface{}) {
	f := v.(script.ScalarFunction)
	p.setRegionsFunc(0, NREGION, f)
}

func (p *ScalarInput) Set(v float64) {
	p.setRegions(0, NREGION, []float64{v})
}

func (p *ScalarInput) setRegionsFunc(r1, r2 int, f script.ScalarFunction) {
	if Const(f) {
		p.setRegions(r1, r2, []float64{f.Float()})
	} else {
		f := f.Fix() // fix values of all variables except t
		p.setFunc(r1, r2, func() []float64 {
			return []float64{f.Eval().(script.ScalarFunction).Float()}
		})
	}
}

func (p *ScalarInput) GetRegion(region int) float64 {
	return float64(p.getRegion(region)[0])
}

func (p *ScalarInput) Eval() interface{}       { return p }
func (p *ScalarInput) Type() reflect.Type      { return reflect.TypeOf(new(ScalarInput)) }
func (p *ScalarInput) InputType() reflect.Type { return script.ScalarFunction_t }
func (p *ScalarInput) Average() float64        { return qAverageUniverse(p)[0] }
func (p *ScalarInput) Region(r int) *sOneReg   { return sOneRegion(p, r) }

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

// these methods should only be accesible from Go

func (p *ScalarInput) SetRegionValueGo(region int, v float64) {
	if region == -1 {
		p.setRegions(0, NREGION, []float64{v})
	} else {
		p.setRegions(region, region+1, []float64{v})
	}
}

func (p *ScalarInput) SetRegionFuncGo(region int, f func() float64) {
	if region == -1 {
		p.setFunc(0, NREGION, func() []float64 {
			return []float64{f()}
		})
	} else {
		p.setFunc(region, region+1, func() []float64 {
			return []float64{f()}
		})
	}
}
