package script

import (
	"go/token"
	"reflect"
)

// converts in to an expression of type OutT.
// pos is used for error message on impossible conversion.
func typeConv(pos token.Pos, in Expr, outT reflect.Type) Expr {
	inT := in.Type()
	switch {
	default:
		panic(err(pos, "type mismatch: can not use type", inT, "as", outT))
	// treat 'void' (type nil) separately:
	case inT == nil && outT != nil:
		panic(err(pos, "void used as value"))
	case inT != nil && outT == nil:
		panic("script internal bug: void input type")
	// strict go conversions:
	case inT == outT:
		return in
	case inT.AssignableTo(outT):
		return in
	// extra conversions for ease-of-use:
	case outT == float64_t && inT == int_t: // int -> float64
		return &intToFloat64{in}
	case inT == float64_t && outT == func_float64_t: // float64 -> func()float64
		return &float64ToFunc{in}
	case inT == int_t && outT == func_float64_t: // int -> func()float64
		return &float64ToFunc{&intToFloat64{in}}
	case inT == vector_t && outT == func_vector_t: // [3]float64 -> func()[3]float64
		return &vectorToFunc{in}
	}
}

// returns input type for expression. Usually this is the same as the return type,
// unless the expression has a method InputType()reflect.Type.
func inputType(e Expr) reflect.Type {
	if in, ok := e.(interface {
		InputType() reflect.Type
	}); ok {
		return in.InputType()
	}
	return e.Type()
}

// common type definitions
var (
	float64_t      = reflect.TypeOf(float64(0))
	func_float64_t = reflect.TypeOf(func() float64 { return 0 })
	int_t          = reflect.TypeOf(int(0))
	string_t       = reflect.TypeOf("")
	vector_t       = reflect.TypeOf([3]float64{})
	func_vector_t  = reflect.TypeOf(func() [3]float64 { panic(0) })
)

// converts int to float64
type intToFloat64 struct{ in Expr }

func (c *intToFloat64) Eval() interface{}  { return float64(c.in.Eval().(int)) }
func (c *intToFloat64) Type() reflect.Type { return float64_t }

// converts float64 to func()float64
type float64ToFunc struct{ in Expr }

func (c *float64ToFunc) Eval() interface{}  { return func() float64 { return c.in.Eval().(float64) } }
func (c *float64ToFunc) Type() reflect.Type { return func_float64_t }

type vectorToFunc struct{ in Expr }

func (c *vectorToFunc) Eval() interface{} {
	return func() [3]float64 { return c.in.Eval().([3]float64) }
}
func (c *vectorToFunc) Type() reflect.Type { return func_vector_t }
