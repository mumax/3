package engine

import (
	"code.google.com/p/mx3/script"
	"log"
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
	world.Var("aex", &Aex)
	world.Var("msat", &Msat)
	world.Var("alpha", &Alpha)
	world.Var("b_ext", &B_ext)
	world.Var("dmi", &DMI)
	world.Var("ku1", &Ku1)
	world.Var("xi", &Xi)
	world.Var("spinpol", &SpinPol)
	world.Var("j", &J)
	world.Var("m", &M)
}

func Compile(src string) (script.Stmt, error) {
	return world.Compile(src)
}

//// needed only to make it callable from scripts
//func (f *ScalFn) Eval() interface{} {
//	return (*f)()
//}
//
//// needed only to make it callable from scripts
//func (f *ScalFn) Assign(e script.Expr) {
//	(*f) = func() float64 { return e.Eval().(float64) }
//}
//
//// needed only to make it callable from scripts
//func (f *VecFn) Eval() interface{} {
//	return (*f)()
//}
//
//// needed only to make it callable from scripts
//func (f *VecFn) Assign(e script.Expr) {
//	(*f) = func() [3]float64 {
//		return eval3float64(e)
//	}
//}
//
//// evaluate e and cast return value to [3]float64 (vector)
//func eval3float64(e script.Expr) [3]float64 {
//	v := e.Eval().([]interface{})
//	util.Argument(len(v) == 3)
//	return [3]float64{v[0].(float64), v[1].(float64), v[2].(float64)}
//
//}
//
//// needed only to make it callable from scripts
//func (b *buffered) Assign(e script.Expr) {
//	b.Set(e.Eval().(*data.Slice))
//}
//
//// needed only to make it callable from scripts
//func (b *buffered) Eval() interface{} {
//	return b
//}

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
