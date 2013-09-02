package engine

// support for interpreted input scripts

import (
	"code.google.com/p/mx3/script"
	"log"
	"math"
)

var World = script.NewWorld()

func init() {
	World.Func("vector", Vector, "Constructs a vector with given components")
	World.Func("average", Average)
	World.Const("mu0", Mu0, "Permittivity of vaccum (Tm/A)")
	World.Func("expect", expect, "Used internally for automated tests: checks if a value is close enough to the expected value")
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

func Compile(src string) (script.Expr, error) {
	World.EnterScope() // file-level scope
	defer World.ExitScope()
	return World.Compile(src)
}

func Vector(x, y, z float64) [3]float64 {
	return [3]float64{x, y, z}
}
