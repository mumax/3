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

	world.Func("autosave", doAutosave)
	world.Func("savetable", doSaveTable)

	world.Func("average", average)

	world.Var("t", &Time)
	world.Var("aex", &Aex)
	world.Var("msat", &Msat)
	world.Var("alpha", &Alpha)
	world.Var("b_ext", &B_ext)
	world.Var("dmi", &DMI)
	world.Var("ku1", &Ku1)
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
	world.ROnly("B_demag", &B_demag)

	fftmAddr := &FFTM
	world.Var("FFTm", &fftmAddr)

	world.LValue("ExchangeMask", &ExchangeMask)

	world.Func("expect", expect)
}

// Test if have lies within want +/- maxError,
// and print suited message.
func expect(msg string, have, want, maxError float64) {
	if math.Abs(have-want) > maxError {
		log.Fatal(msg, ":", " have:", have, " want:", want, "±", maxError)
	} else {
		log.Println(msg, ":", have, "OK")
	}
}

func Compile(src string) (script.Expr, error) {
	return world.Compile(src)
}

// needed only to make it callable from scripts
func (b *bufferedQuant) SetValue(v interface{})  { b.Set(v.(*data.Slice)) }
func (b *bufferedQuant) Eval() interface{}       { return b }
func (b *bufferedQuant) Type() reflect.Type      { return reflect.TypeOf(new(bufferedQuant)) }
func (b *bufferedQuant) InputType() reflect.Type { return reflect.TypeOf(new(data.Slice)) }

// needed only to make it callable from scripts
func (b *staggeredMaskQuant) SetValue(v interface{})  { b.Set(v.(*data.Slice)) }
func (b *staggeredMaskQuant) Eval() interface{}       { return b }
func (b *staggeredMaskQuant) Type() reflect.Type      { return reflect.TypeOf(new(staggeredMaskQuant)) }
func (b *staggeredMaskQuant) InputType() reflect.Type { return reflect.TypeOf(new(data.Slice)) }

func doAutosave(what interface {
	Autosave(float64)
}, period float64) {
	what.Autosave(period)
}

func doSaveTable(period float64) {
	Table.Autosave(period)
}

func Vector(x, y, z float64) [3]float64 {
	return [3]float64{x, y, z}
}
