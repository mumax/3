package engine

// support for interpreted input scripts

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/script"
	"log"
	"math"
	"reflect"
)

var world = script.NewWorld()

func init() {
	world.Func("setgridsize", setGridSize)
	world.Func("setcellsize", setCellSize)

	world.Func("vector", Vector)

	world.Func("run", Run)
	world.Func("steps", Steps)

	world.Func("savetable", doSaveTable)

	world.Func("average", Average)

	world.Var("t", &Time)
	world.Var("aex", &Aex)
	world.Var("msat", &Msat)
	world.Var("alpha", &Alpha)
	world.Var("b_ext", &B_ext)
	world.Var("dmi", &DMI)
	world.Var("xi", &Xi)
	world.Var("spinpol", &SpinPol)
	world.Var("j", &J)

	world.Var("EnableDemag", &EnableDemag)

	world.Var("Dt", &Solver.Dt_si)
	world.Var("MinDt", &Solver.Mindt)
	world.Var("MaxDt", &Solver.Maxdt)
	world.Var("MaxErr", &Solver.MaxErr)
	world.Var("Headroom", &Solver.Headroom)
	world.Var("FixDt", &Solver.Fixdt)

	world.Const("mu0", Mu0)

	world.LValue("m", &M)
	world.Func("SetGeom", SetGeometry)
	world.Func("DefRegion", DefRegion)

	B_demag_addr := &B_demag
	world.ROnly("B_demag", &B_demag_addr)
	B_exch_addr := &B_exch
	world.ROnly("B_exch", &B_exch_addr)
	B_uni_addr := &B_uni
	world.ROnly("B_uni", &B_uni_addr)
	B_dmi_addr := &B_dmi
	world.ROnly("B_dmi", &B_dmi_addr)
	fftm_addr := &FFTM
	world.ROnly("mFFT", &fftm_addr)
	regions_addr := &regions
	world.ROnly("regions", &regions_addr)
	Ku1_addr := &Ku1
	world.ROnly("Ku1", &Ku1_addr)

	world.LValue("ExchangeMask", &ExchangeMask)
	world.LValue("AnisotropyMask", &KuMask)

	world.Func("expect", expect)
}

// Test if have lies within want +/- maxError,
// and print suited message.
func expect(msg string, have, want, maxError float64) {
	if math.Abs(have-want) > maxError {
		log.Fatal(msg, ":", " have: ", have, " want: ", want, "±", maxError)
	} else {
		log.Println(msg, ":", have, "OK")
	}
}

func Compile(src string) (script.Expr, error) {
	world.EnterScope() // file-level scope
	defer world.ExitScope()
	return world.Compile(src)
}

func (m *magnetization) SetValue(v interface{})  { m.setRegion(v.(Config), nil) }
func (m *magnetization) InputType() reflect.Type { return reflect.TypeOf(Config(nil)) }

func (b *bufferedQuant) SetValue(v interface{})  { b.Set(v.(*data.Slice)) }
func (b *bufferedQuant) Eval() interface{}       { return b }
func (b *bufferedQuant) Type() reflect.Type      { return reflect.TypeOf(new(bufferedQuant)) }
func (b *bufferedQuant) InputType() reflect.Type { return reflect.TypeOf(new(data.Slice)) }

// needed only to make it callable from scripts
func (b *maskQuant) SetValue(v interface{}) { b.Set(v.(*data.Slice)) }
func (b *maskQuant) Type() reflect.Type     { return reflect.TypeOf(new(maskQuant)) }

// needed only to make it callable from scripts
func (b *staggeredMaskQuant) SetValue(v interface{})  { b.Set(v.(*data.Slice)) }
func (b *staggeredMaskQuant) Eval() interface{}       { return b }
func (b *staggeredMaskQuant) Type() reflect.Type      { return reflect.TypeOf(new(staggeredMaskQuant)) }
func (b *staggeredMaskQuant) InputType() reflect.Type { return reflect.TypeOf(new(data.Slice)) }

func doSaveTable(period float64) {
	Table.Autosave(period)
}

func Vector(x, y, z float64) [3]float64 {
	return [3]float64{x, y, z}
}
