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
	world.Func("vector", Vector)
	world.Func("average", Average)
	world.Const("mu0", Mu0)
	world.Func("expect", expect)
}

// Test if have lies within want +/- maxError,
// and print suited message.
func expect(msg string, have, want, maxError float64) {
	if math.IsNaN(have) || math.IsNaN(want) || math.Abs(have-want) > maxError {
		log.Fatal(msg, ":", " have: ", have, " want: ", want, "±", maxError)
	} else {
		log.Println(msg, ":", have, "OK")
	}
	// note: also check "want" for NaN in case "have" and "want" are switched.
}

func Compile(src string) (script.Expr, error) {
	world.EnterScope() // file-level scope
	defer world.ExitScope()
	return world.Compile(src)
}

func (b *bufferedQuant) SetValue(v interface{}) { b.Set(v.(*data.Slice)) }
func (b *bufferedQuant) Eval() interface{}      { return b }

//func (b *bufferedQuant) Type() reflect.Type      { return reflect.TypeOf(new(bufferedQuant)) }
func (b *bufferedQuant) InputType() reflect.Type { return reflect.TypeOf(new(data.Slice)) }

func Vector(x, y, z float64) [3]float64 {
	return [3]float64{x, y, z}
}
