package engine

// support for interpreted input scripts

import (
	"code.google.com/p/mx3/mag"
	"code.google.com/p/mx3/script"
)

var world = script.NewWorld()

func init() {
	DeclFunc("vector", Vector, "Constructs a vector with given components")
	DeclFunc("average", Average, "Average of space-dependent quantity")
	DeclConst("mu0", mag.Mu0, "Permittivity of vaccum (Tm/A)")
}

func DeclFunc(name string, f interface{}, doc string) {
	world.Func(name, f, doc)
}

func DeclConst(name string, value float64, doc string) {
	world.Const(name, value, doc)
}

func DeclVar(name string, value interface{}, doc string) {
	world.Var(name, value, doc)
	guiAdd(name, value)
}

func DeclLValue(name string, value script.LValue, doc string) {
	world.LValue(name, value, doc)
	guiAdd(name, value)
}

func DeclROnly(name string, value interface{}, doc string) {
	world.ROnly(name, value, doc)
	guiAdd(name, value)
}

func guiAdd(name string, value interface{}) {
	if v, ok := value.(Param); ok {
		params[name] = v
	}
	if v, ok := value.(Getter); ok {
		quants[name] = v
	}
}

func Compile(src string) (script.Expr, error) {
	world.EnterScope() // file-level scope
	defer world.ExitScope()
	return world.Compile(src)
}
