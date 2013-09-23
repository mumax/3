package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/script"
	"github.com/mumax/3/util"
	"reflect"
)

// An excitation, typically field or current,
// can be defined region-wise plus extra mask*multiplier terms.
type excitation struct {
	perRegion  VectorParam // Region-based excitation
	extraTerms []mulmask   // add extra mask*multiplier terms
}

type mulmask struct {
	mul  func() float64
	mask *data.Slice
}

func (e *excitation) init(name, unit, desc string) {
	e.perRegion.init(name+"_perRegion", unit, "(internal)")
	DeclLValue(name, e, desc)
}

func (e *excitation) addTo(dst *data.Slice) {
	if !e.perRegion.isZero() {
		cuda.RegionAddV(dst, e.perRegion.LUT(), regions.Gpu())
	}
	for _, t := range e.extraTerms {
		var mul float32 = 1
		if t.mul != nil {
			mul = float32(t.mul())
		}
		cuda.Madd2(dst, dst, t.mask, 1, mul)
	}
}

func (e *excitation) isZero() bool {
	return e.perRegion.isZero() && len(e.extraTerms) == 0
}

func (e *excitation) Get() (*data.Slice, bool) {
	buf := cuda.Buffer(e.NComp(), e.Mesh())
	cuda.Zero(buf)
	e.addTo(buf)
	return buf, true
}

// Add an extra maks*multiplier term to the excitation.
// TODO: unittest!
func (e *excitation) Add(mask *data.Slice, mul func() float64) {
	e.extraTerms = append(e.extraTerms, mulmask{mul, assureGPU(mask)})
}

// TODO: EvalExpr() to be used everywhere?
func (e *excitation) Ext_AddExpr(expr string, mul script.ScalarFunction) {
	World.EnterScope()
	defer World.ExitScope()

	var x, y, z float64
	World.Var("x", &x)
	World.Var("y", &y)
	World.Var("z", &z)
	f, err := World.CompileExpr(expr)
	util.FatalErr(err)

	host := data.NewSlice(3, e.Mesh())
	h := host.Vectors()
	n := e.Mesh().Size()
	c := e.Mesh().CellSize()
	dx := (float64(n[2]/2) - 0.5) * c[2]
	dy := (float64(n[1]/2) - 0.5) * c[1]
	dz := (float64(n[0]/2) - 0.5) * c[0]

	for i := 0; i < n[0]; i++ {
		z = float64(i)*c[0] - dz
		for j := 0; j < n[1]; j++ {
			y = float64(j)*c[1] - dy
			for k := 0; k < n[2]; k++ {
				x = float64(k)*c[2] - dx
				v := f.Eval().([3]float64)
				h[0][i][j][k] = float32(v[0])
				h[1][i][j][k] = float32(v[1])
				h[2][i][j][k] = float32(v[2])
			}
		}
	}
	e.Add(cuda.GPUCopy(host), func() float64 { return mul.Float() })
}

func assureGPU(s *data.Slice) *data.Slice {
	if s.GPUAccess() {
		return s
	} else {
		return cuda.GPUCopy(s)
	}
}

// user script: has to be 3-vector
func (e *excitation) SetRegion(region int, value [3]float64) {
	e.perRegion.setRegion(region, value[:])
}

// for gui (nComp agnostic)
func (e *excitation) setRegion(region int, value []float64) {
	e.perRegion.setRegion(region, value)
}

// does not use extramask!
func (e *excitation) getRegion(region int) []float64 {
	return e.perRegion.getRegion(region)
}

func (e *excitation) TableData() []float64 {
	return e.perRegion.getRegion(0)
}

func (p *excitation) Region(r int) TableData {
	return p.perRegion.Region(r)
}

// needed for script

func (e *excitation) SetValue(v interface{}) {
	e.perRegion.SetValue(v) // allows function of time
}

func (e *excitation) Name() string            { return e.perRegion.Name() }
func (e *excitation) Unit() string            { return e.perRegion.Unit() }
func (e *excitation) NComp() int              { return e.perRegion.NComp() }
func (e *excitation) Mesh() *data.Mesh        { return &globalmesh }
func (e *excitation) Eval() interface{}       { return e }
func (e *excitation) Type() reflect.Type      { return reflect.TypeOf(new(excitation)) }
func (e *excitation) InputType() reflect.Type { return script.VectorFunction_t }
