package engine

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/script"
	"log"
	"reflect"
)

var world = script.NewWorld()

func init() {
	world.Func("print", myprint)
	world.Func("setgridsize", setGridSize)
	world.Func("setcellsize", setCellSize)
	world.Func("run", Run)
	world.Func("steps", Steps)
	world.Func("autosave", doAutosave)
	world.Func("savetable", doSaveTable)
	world.Func("average", average)
	world.Var("t", &Time)
	world.LValue("aex", &Aex)
	world.LValue("msat", &Msat)
	world.LValue("alpha", &Alpha)
	world.Var("b_ext", &B_ext)
	world.Var("dmi", &DMI)
	world.Var("ku1", &Ku1)
	world.Var("xi", &Xi)
	world.Var("spinpol", &SpinPol)
	world.Var("j", &J)
	world.LValue("m", (*bufL)(&M))
}

func Compile(src string) (script.Expr, error) {
	return world.Compile(src)
}

// needed only to make it callable from scripts
func (f *ScalFn) Eval() interface{} {
	return (*f)()
}

// needed only to make it callable from scripts
func (f *ScalFn) Set(v interface{}) {
	(*f) = func() float64 { return v.(float64) }
}

func (f *ScalFn) Type() reflect.Type {
	return reflect.TypeOf(float64(0))
}

// needed only to make it callable from scripts
func (f *VecFn) Eval() interface{} {
	return (*f)()
}

// needed only to make it callable from scripts
func (f *VecFn) Set(v interface{}) {
	(*f) = func() [3]float64 {
		return v.([3]float64)
	}
}

func (f *VecFn) Type() reflect.Type {
	return reflect.TypeOf([3]float64{})
}

// evaluate e and cast return value to [3]float64 (vector)
//func eval3float64(e script.Expr) [3]float64 {
//	v := e.Eval().([]interface{})
//	util.Argument(len(v) == 3)
//	return [3]float64{v[0].(float64), v[1].(float64), v[2].(float64)}
//
//}

type bufL buffered

// needed only to make it callable from scripts
func (b *bufL) Set(v interface{}) {
	b.Set(v.(*data.Slice))
}

// needed only to make it callable from scripts
func (b *bufL) Eval() interface{} {
	return b
}

func (b *bufL) Type() reflect.Type {
	return reflect.TypeOf((*buffered)(b))
}

func myprint(msg ...interface{}) {
	log.Println(msg...)
}

func doAutosave(what interface {
	Autosave(float64)
}, period float64) {
	what.Autosave(period)
}

func doSaveTable(period float64) {
	Table.Autosave(period)
}
