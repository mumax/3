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
	world.Func("savetable", doSaveTable)
	world.Func("average", Average)
	world.Const("mu0", Mu0)
	world.LValue("m", &M)
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

func (m *magnetization) SetValue(v interface{})  { m.setRegion(v.(Config), nil) }
func (m *magnetization) InputType() reflect.Type { return reflect.TypeOf(Config(nil)) }

func (b *bufferedQuant) SetValue(v interface{})  { b.Set(v.(*data.Slice)) }
func (b *bufferedQuant) Eval() interface{}       { return b }
func (b *bufferedQuant) Type() reflect.Type      { return reflect.TypeOf(new(bufferedQuant)) }
func (b *bufferedQuant) InputType() reflect.Type { return reflect.TypeOf(new(data.Slice)) }

// needed only to make it callable from scripts
func (b *maskQuant) SetValue(v interface{}) { b.Set(v.(*data.Slice)) }
func (b *maskQuant) Type() reflect.Type     { return reflect.TypeOf(new(maskQuant)) }

// needed only to make it callable from scripts
func (b *staggeredMaskQuant) SetValue(v interface{})  { b.Set(v.(*data.Slice)) }
func (b *staggeredMaskQuant) Eval() interface{}       { return b }
func (b *staggeredMaskQuant) Type() reflect.Type      { return reflect.TypeOf(new(staggeredMaskQuant)) }
func (b *staggeredMaskQuant) InputType() reflect.Type { return reflect.TypeOf(new(data.Slice)) }

func doSaveTable(period float64) {
	Table.Autosave(period)
}

func Vector(x, y, z float64) [3]float64 {
	return [3]float64{x, y, z}
}
