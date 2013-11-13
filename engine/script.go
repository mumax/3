package engine

// support for interpreted input scripts

import "github.com/mumax/3/script"

var World = script.NewWorld()

func DeclFunc(name string, f interface{}, doc string) {
	World.Func(name, f, doc)
}

func DeclPure(name string, f interface{}, doc string) {
	World.PureFunc(name, f, doc)
}

func DeclConst(name string, value float64, doc string) {
	World.Const(name, value, doc)
}

func DeclVar(name string, value interface{}, doc string) {
	World.Var(name, value, doc)
	guiAdd(name, value)
}

func DeclLValue(name string, value script.LValue, doc string) {
	World.LValue(name, value, doc)
	guiAdd(name, value)
}

func DeclROnly(name string, value interface{}, doc string) {
	World.ROnly(name, value, doc)
	guiAdd(name, value)
}

var history string

// evaluate code, exit on error (behavior for input files)
func EvalFile(code *script.BlockStmt) {
	for i := range code.Children {
		history += script.Format(code.Node[i]) + "<br/>"
		code.Children[i].Eval()
	}
}

//func EvalGUI(code*script.BlockStmt){
//
//}

func guiAdd(name string, value interface{}) {
	if v, ok := value.(Param); ok {
		params[name] = v
	}
	if v, ok := value.(Slicer); ok {
		quants[name] = v
	}
}
