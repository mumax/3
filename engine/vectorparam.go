package engine

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/script"
	"reflect"
	"strings"
)

// vector input parameter, settable by user
type VectorParam struct {
	inputParam
}

func (p *VectorParam) init(name, unit, desc string) {
	p.inputParam.init(VECTOR, name, unit, nil) // no vec param has children (yet)
	if !strings.HasPrefix(name, "_") {         // don't export names beginning with "_" (e.g. from exciation)
		DeclLValue(name, p, cat(desc, unit))
	}
}

func (p *VectorParam) SetRegion(region int, f script.VectorFunction) {
	if region == -1 {
		p.setRegionsFunc(0, NREGION, f) //uniform
	} else {
		p.setRegionsFunc(region, region+1, f)
	}
}

func (p *VectorParam) SetValue(v interface{}) {
	f := v.(script.VectorFunction)
	p.setRegionsFunc(0, NREGION, f)
}

func (p *VectorParam) setRegionsFunc(r1, r2 int, f script.VectorFunction) {
	if Const(f) {
		p.setRegions(r1, r2, slice(f.Float3()))
	} else {
		f := f.Fix() // fix values of all variables except t
		p.setFunc(r1, r2, func() []float64 {
			return slice(f.Eval().(script.VectorFunction).Float3())
		})
	}
}

func (p *VectorParam) SetRegionFn(region int, f func() [3]float64) {
	p.setFunc(region, region+1, func() []float64 {
		return slice(f())
	})
}

func (p *VectorParam) GetRegion(region int) [3]float64 {
	v := p.getRegion(region)
	return unslice(v)
}

func (p *VectorParam) Eval() interface{}       { return p }
func (p *VectorParam) Type() reflect.Type      { return reflect.TypeOf(new(VectorParam)) }
func (p *VectorParam) InputType() reflect.Type { return script.VectorFunction_t }
func (p *VectorParam) Region(r int) *vOneReg   { return vOneRegion(p, r) }
func (p *VectorParam) Average() data.Vector    { return unslice(qAverageUniverse(p)) }
func (p *VectorParam) Comp(c int) *comp        { return Comp(p, c) }
