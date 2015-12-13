package engine

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/script"
	"reflect"
	"strings"
)

// vector input parameter, settable by user
type VectorInput struct {
	inputParam
}

func (p *VectorInput) init(name, unit, desc string) {
	p.inputParam.init(VECTOR, name, unit, nil) // no vec param has children (yet)
	if !strings.HasPrefix(name, "_") {         // don't export names beginning with "_" (e.g. from exciation)
		DeclLValue(name, p, cat(desc, unit))
	}
}

func (p *VectorInput) SetRegion(region int, f script.VectorFunction) {
	if region == -1 {
		p.setRegionsFunc(0, NREGION, f) //uniform
	} else {
		p.setRegionsFunc(region, region+1, f)
	}
}

func (p *VectorInput) SetValue(v interface{}) {
	f := v.(script.VectorFunction)
	p.setRegionsFunc(0, NREGION, f)
}

func (p *VectorInput) setRegionsFunc(r1, r2 int, f script.VectorFunction) {
	if Const(f) {
		p.setRegions(r1, r2, slice(f.Float3()))
	} else {
		f := f.Fix() // fix values of all variables except t
		p.setFunc(r1, r2, func() []float64 {
			return slice(f.Eval().(script.VectorFunction).Float3())
		})
	}
}

func (p *VectorInput) SetRegionFn(region int, f func() [3]float64) {
	p.setFunc(region, region+1, func() []float64 {
		return slice(f())
	})
}

func (p *VectorInput) GetRegion(region int) [3]float64 {
	v := p.getRegion(region)
	return unslice(v)
}

func (p *VectorInput) Eval() interface{}       { return p }
func (p *VectorInput) Type() reflect.Type      { return reflect.TypeOf(new(VectorInput)) }
func (p *VectorInput) InputType() reflect.Type { return script.VectorFunction_t }
func (p *VectorInput) Region(r int) *vOneReg   { return vOneRegion(p, r) }
func (p *VectorInput) Average() data.Vector    { return unslice(qAverageUniverse(p)) }
func (p *VectorInput) Comp(c int) ScalarField  { return Comp(p, c) }
