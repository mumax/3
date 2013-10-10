package engine

import (
	"github.com/mumax/3/script"
	"github.com/mumax/3/util"
	"log"
	"reflect"
)

type VectorParam struct {
	inputParam
}

func (p *VectorParam) init(name, unit, desc string) {
	p.inputParam.init(3, name, unit, nil) // no vec param has children (yet)
	DeclLValue(name, p, cat(desc, unit))
}

func (p *VectorParam) SetRegion(region int, f script.VectorFunction) {
	p.setRegionsFunc(region, region+1, f)
}

func (p *VectorParam) SetValue(v interface{}) {
	f := v.(script.VectorFunction)
	p.setRegionsFunc(0, NREGION, f)
}

func (p *VectorParam) setRegionsFunc(r1, r2 int, f script.VectorFunction) {
	if Const(f) {
		log.Println(p.Name(), "[", r1, ":", r2, "]", "is constant")
		p.setRegions(r1, r2, slice(f.Float3()))
	} else {
		log.Println(p.Name(), "[", r1, ":", r2, "]", "is not constant")
		p.setFunc(r1, r2, func() []float64 {
			return slice(f.Float3())
		})
	}
}

func (p *VectorParam) GetRegion(region int) [3]float64 {
	v := p.getRegion(region)
	return unslice(v)
}

func (p *VectorParam) Eval() interface{}       { return p }
func (p *VectorParam) Type() reflect.Type      { return reflect.TypeOf(new(VectorParam)) }
func (p *VectorParam) InputType() reflect.Type { return script.VectorFunction_t }

// shortcut for slicing unaddressable_vector()[:]
func slice(v [3]float64) []float64 {
	return v[:]
}

func unslice(v []float64) [3]float64 {
	util.Assert(len(v) == 3)
	return [3]float64{v[0], v[1], v[2]}
}
