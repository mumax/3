package engine

// support for interpreted input scripts

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/script"
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
	world.LValue("m", &M)
}

func Compile(src string) (script.Expr, error) {
	return world.Compile(src)
}

// needed only to make it callable from scripts
func (b *buffered) SetValue(v interface{})  { b.Set(v.(*data.Slice)) }
func (b *buffered) Eval() interface{}       { return b }
func (b *buffered) Type() reflect.Type      { return reflect.TypeOf((*buffered)(b)) }
func (b *buffered) InputType() reflect.Type { return reflect.TypeOf(new(data.Slice)) }

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
