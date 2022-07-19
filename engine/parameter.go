package engine

/*
parameters are region- and time dependent input values,
like material parameters.
*/

import (
	"fmt"
	"math"
	"reflect"
	"strings"

	"github.com/mumax/3/v3/cuda"
	"github.com/mumax/3/v3/data"
	"github.com/mumax/3/v3/script"
	"github.com/mumax/3/v3/util"
)

// input parameter, settable by user
type regionwise struct {
	lut
	upd_reg    [NREGION]func() []float64 // time-dependent values
	timestamp  float64                   // used not to double-evaluate f(t)
	children   []derived                 // derived parameters
	name, unit string
}

func (p *regionwise) init(nComp int, name, unit string, children []derived) {
	p.lut.init(nComp, p)
	p.name = name
	p.unit = unit
	p.children = children
	p.timestamp = math.Inf(-1)
}

func (p *regionwise) MSlice() cuda.MSlice {
	if p.IsUniform() {
		return cuda.MakeMSlice(data.NilSlice(p.NComp(), Mesh().Size()), p.getRegion(0))
	} else {
		buf, r := p.Slice()
		util.Assert(r == true)
		return cuda.ToMSlice(buf)
	}
}

func (p *regionwise) Name() string     { return p.name }
func (p *regionwise) Unit() string     { return p.unit }
func (p *regionwise) Mesh() *data.Mesh { return Mesh() }

func (p *regionwise) addChild(c ...derived) {
	for _, c := range c {
		// TODO: no duplicates
		if !contains(p.children, c) {
			p.children = append(p.children, c)
			fmt.Println(p, ".addChild", c)
		}
	}
}

func contains(s []derived, x derived) bool {
	for _, y := range s {
		if y == x {
			return true
		}
	}
	return false
}

func (p *regionwise) update() {
	if p.timestamp != Time {
		changed := false
		// update functions of time
		for r := 0; r < NREGION; r++ {
			updFunc := p.upd_reg[r]
			if updFunc != nil {
				p.bufset_(r, updFunc())
				changed = true
			}
		}
		p.timestamp = Time
		if changed {
			p.invalidate()
		}
	}
}

// set in one region
func (p *regionwise) setRegion(region int, v []float64) {
	if region == -1 {
		p.setUniform(v)
	} else {
		p.setRegions(region, region+1, v)
	}
}

// set in all regions
func (p *regionwise) setUniform(v []float64) {
	p.setRegions(0, NREGION, v)
}

// set in regions r1..r2(excl)
func (p *regionwise) setRegions(r1, r2 int, v []float64) {
	util.Argument(len(v) == len(p.cpu_buf))
	util.Argument(r1 < r2) // exclusive upper bound
	for r := r1; r < r2; r++ {
		p.upd_reg[r] = nil
		p.bufset_(r, v)
	}
	p.invalidate()
}

func (p *regionwise) bufset_(region int, v []float64) {
	for c := range p.cpu_buf {
		p.cpu_buf[c][region] = float32(v[c])
	}
}

func (p *regionwise) setFunc(r1, r2 int, f func() []float64) {
	util.Argument(r1 < r2) // exclusive upper bound
	for r := r1; r < r2; r++ {
		p.upd_reg[r] = f
	}
	p.invalidate()
}

// mark my GPU copy and my children as invalid (need update)
func (p *regionwise) invalidate() {
	p.gpu_ok = false
	for _, c := range p.children {
		c.invalidate()
	}
}

func (p *regionwise) getRegion(region int) []float64 {
	cpu := p.cpuLUT()
	v := make([]float64, p.NComp())
	for i := range v {
		v[i] = float64(cpu[i][region])
	}
	return v
}

func (p *regionwise) IsUniform() bool {
	cpu := p.cpuLUT()
	v1 := p.getRegion(0)
	for r := 1; r < NREGION; r++ {
		for c := range v1 {
			if cpu[c][r] != float32(v1[c]) {
				return false
			}
		}
	}
	return true
}

func (p *regionwise) average() []float64 { return qAverageUniverse(p) }

// parameter derived from others (not directly settable). E.g.: Bsat derived from Msat
type DerivedParam struct {
	lut                          // GPU storage
	updater  func(*DerivedParam) // called to update my value
	uptodate bool                // cleared if parents' value change
	parents  []updater           // parents updated before I'm updated
}

// any parameter that depends on an inputParam
type derived interface {
	invalidate()
}

type parent interface {
	update()
	addChild(...derived)
}

func NewDerivedParam(nComp int, parents []parent, updater func(*DerivedParam)) *DerivedParam {
	p := new(DerivedParam)
	p.lut.init(nComp, p) // pass myself to update me if needed
	p.updater = updater
	for _, P := range parents {
		p.parents = append(p.parents, P)
	}
	return p
}

func (d *DerivedParam) init(nComp int, parents []parent, updater func(*DerivedParam)) {
	d.lut.init(nComp, d) // pass myself to update me if needed
	d.updater = updater
	for _, p := range parents {
		d.parents = append(d.parents, p)
		p.addChild(d)
	}
}

func (p *DerivedParam) invalidate() {
	p.uptodate = false
}

func (p *DerivedParam) update() {
	for _, par := range p.parents {
		par.update() // may invalidate me
	}
	if !p.uptodate {
		p.updater(p)
		p.gpu_ok = false
		p.uptodate = true
	}
}

// Get value in region r.
func (p *DerivedParam) GetRegion(r int) []float64 {
	lut := p.cpuLUT() // updates me if needed
	v := make([]float64, p.NComp())
	for c := range v {
		v[c] = float64(lut[c][r])
	}
	return v
}

// specialized param with 1 component
type RegionwiseScalar struct {
	regionwise
}

func (p *RegionwiseScalar) init(name, unit, desc string, children []derived) {
	p.regionwise.init(SCALAR, name, unit, children)
	if !strings.HasPrefix(name, "_") { // don't export names beginning with "_" (e.g. from exciation)
		DeclLValue(name, p, cat(desc, unit))
	}
}

// TODO: auto derived
func NewScalarParam(name, unit, desc string, children ...derived) *RegionwiseScalar {
	p := new(RegionwiseScalar)
	p.regionwise.init(SCALAR, name, unit, children)
	if !strings.HasPrefix(name, "_") { // don't export names beginning with "_" (e.g. from exciation)
		DeclLValue(name, p, cat(desc, unit))
	}
	return p
}

func (p *RegionwiseScalar) SetRegion(region int, f script.ScalarFunction) {
	if region == -1 {
		p.setRegionsFunc(0, NREGION, f) // uniform
	} else {
		p.setRegionsFunc(region, region+1, f) // upper bound exclusive
	}
}

func (p *RegionwiseScalar) SetValue(v interface{}) {
	f := v.(script.ScalarFunction)
	p.setRegionsFunc(0, NREGION, f)
}

func (p *RegionwiseScalar) Set(v float64) {
	p.setRegions(0, NREGION, []float64{v})
}

func (p *RegionwiseScalar) setRegionsFunc(r1, r2 int, f script.ScalarFunction) {
	if IsConst(f) {
		p.setRegions(r1, r2, []float64{f.Float()})
	} else {
		f := f.Fix() // fix values of all variables except t
		p.setFunc(r1, r2, func() []float64 {
			return []float64{f.Eval().(script.ScalarFunction).Float()}
		})
	}
}

func (p *RegionwiseScalar) GetRegion(region int) float64 {
	return float64(p.getRegion(region)[0])
}

func (p *RegionwiseScalar) Eval() interface{}       { return p }
func (p *RegionwiseScalar) Type() reflect.Type      { return reflect.TypeOf(new(RegionwiseScalar)) }
func (p *RegionwiseScalar) InputType() reflect.Type { return script.ScalarFunction_t }
func (p *RegionwiseScalar) Average() float64        { return qAverageUniverse(p)[0] }
func (p *RegionwiseScalar) Region(r int) *sOneReg   { return sOneRegion(p, r) }

// checks if a script expression contains t (time)
func IsConst(e script.Expr) bool {
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

func (p *RegionwiseScalar) SetRegionValueGo(region int, v float64) {
	if region == -1 {
		p.setRegions(0, NREGION, []float64{v})
	} else {
		p.setRegions(region, region+1, []float64{v})
	}
}

func (p *RegionwiseScalar) SetRegionFuncGo(region int, f func() float64) {
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

// vector input parameter, settable by user
type RegionwiseVector struct {
	regionwise
}

func NewVectorParam(name, unit, desc string) *RegionwiseVector {
	p := new(RegionwiseVector)
	p.regionwise.init(VECTOR, name, unit, nil) // no vec param has children (yet)
	if !strings.HasPrefix(name, "_") {         // don't export names beginning with "_" (e.g. from exciation)
		DeclLValue(name, p, cat(desc, unit))
	}
	return p
}

func (p *RegionwiseVector) SetRegion(region int, f script.VectorFunction) {
	if region == -1 {
		p.setRegionsFunc(0, NREGION, f) //uniform
	} else {
		p.setRegionsFunc(region, region+1, f)
	}
}

func (p *RegionwiseVector) SetValue(v interface{}) {
	f := v.(script.VectorFunction)
	p.setRegionsFunc(0, NREGION, f)
}

func (p *RegionwiseVector) setRegionsFunc(r1, r2 int, f script.VectorFunction) {
	if IsConst(f) {
		p.setRegions(r1, r2, slice(f.Float3()))
	} else {
		f := f.Fix() // fix values of all variables except t
		p.setFunc(r1, r2, func() []float64 {
			return slice(f.Eval().(script.VectorFunction).Float3())
		})
	}
}

func (p *RegionwiseVector) SetRegionFn(region int, f func() [3]float64) {
	p.setFunc(region, region+1, func() []float64 {
		return slice(f())
	})
}

func (p *RegionwiseVector) GetRegion(region int) [3]float64 {
	v := p.getRegion(region)
	return unslice(v)
}

func (p *RegionwiseVector) Eval() interface{}       { return p }
func (p *RegionwiseVector) Type() reflect.Type      { return reflect.TypeOf(new(RegionwiseVector)) }
func (p *RegionwiseVector) InputType() reflect.Type { return script.VectorFunction_t }
func (p *RegionwiseVector) Region(r int) *vOneReg   { return vOneRegion(p, r) }
func (p *RegionwiseVector) Average() data.Vector    { return unslice(qAverageUniverse(p)) }
func (p *RegionwiseVector) Comp(c int) ScalarField  { return Comp(p, c) }
