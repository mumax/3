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

	world.Var("dt", &Solver.Dt_si)
	world.Var("mindt", &Solver.Mindt)
	world.Var("maxdt", &Solver.Maxdt)
	world.Var("maxerr", &Solver.MaxErr)
	world.Var("headroom", &Solver.Headroom)
	world.Var("fixdt", &Solver.Fixdt)

	world.LValue("m", &M)
	world.LValue("ExchangeMask", ExchangeMask)

	world.Func("expect", expect)
}

func expect(msg string, have, want, maxError float64) {
	if math.Abs(have-want) > maxError {
		log.Fatal(msg, ":", "have:", have, "want:", want)
	}
}

func Compile(src string) (script.Expr, error) {
	return world.Compile(src)
}

// needed only to make it callable from scripts
func (b *buffered) SetValue(v interface{})  { b.Set(v.(*data.Slice)) }
func (b *buffered) Eval() interface{}       { return b }
func (b *buffered) Type() reflect.Type      { return reflect.TypeOf(new(buffered)) }
func (b *buffered) InputType() reflect.Type { return reflect.TypeOf(new(data.Slice)) }

// needed only to make it callable from scripts
func (b *StaggeredMask) SetValue(v interface{})  { b.Set(v.(*data.Slice)) }
func (b *StaggeredMask) Eval() interface{}       { return b }
func (b *StaggeredMask) Type() reflect.Type      { return reflect.TypeOf(new(StaggeredMask)) }
func (b *StaggeredMask) InputType() reflect.Type { return reflect.TypeOf(new(data.Slice)) }

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
