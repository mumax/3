package engine

// support for interpreted input scripts

import (
	"code.google.com/p/mx3/script"
	"log"
	"math"
)

var world = script.NewWorld()

func init() {
	DeclFunc("vector", Vector, "Constructs a vector with given components")
	DeclFunc("average", Average, "Average of space-dependent quantity")
	DeclConst("mu0", Mu0, "Permittivity of vaccum (Tm/A)")
	DeclFunc("expect", expect, "Used internally for automated tests: checks if a value is close enough to the expected value")
	DeclFunc("fprintln", Fprintln, "Print to file")
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

// Test if have lies within want +/- maxError,
// and print suited message.
func expect(msg string, have, want, maxError float64) {
	if math.IsNaN(have) || math.IsNaN(want) || math.Abs(have-want) > maxError {
		log.Fatal(msg, ":", " have: ", have, " want: ", want, "±", maxError)
	} else {
		log.Println(msg, ":", have, "OK")
	}
	// note: we also check "want" for NaN in case "have" and "want" are switched.
}
