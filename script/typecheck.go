package script

import (
	"fmt"
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
	// int -> float64
	case outT == float64_t && inT == int_t:
		return &intToFloat64{in}

	// float64 -> int
	case outT == int_t && inT == float64_t:
		return &float64ToInt{in}

	case inT == float64_t && outT.AssignableTo(funcIf_t):
		return &funcIf{in}

	case inT == int_t && outT.AssignableTo(funcIf_t):
		return &funcIf{&intToFloat64{in}}

		// float64 -> func()float64
		//case inT == float64_t && outT == func_float64_t:
		//	return &float64ToFunc{in}

		//// float64 -> func()float64
		//case inT == func_float64_t && outT == float64_t:
		//	return &funcToFloat64{in}

		// int -> func()float64
		//case inT == int_t && outT == func_float64_t:
		//	return &float64ToFunc{&intToFloat64{in}}

		// [3]float64 -> func()[3]float64
		//case inT == vector_t && outT == func_vector_t:
		//	return &vectorToFunc{in}
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
	bool_t         = reflect.TypeOf(false)
	funcIf_t       = reflect.TypeOf(dummy_f).In(0)
)

type FuncIf interface {
	Float() float64
}

// maneuvers to get interface type of Func (simpler way?)
func dummy_f(FuncIf) {}

// converts int to float64
type intToFloat64 struct{ in Expr }

func (c *intToFloat64) Eval() interface{}  { return float64(c.in.Eval().(int)) }
func (c *intToFloat64) Type() reflect.Type { return float64_t }

// converts float64 to int
type float64ToInt struct{ in Expr }

func (c *float64ToInt) Eval() interface{}  { return safe_int(c.in.Eval().(float64)) }
func (c *float64ToInt) Type() reflect.Type { return int_t }
func safe_int(x float64) int {
	i := int(x)
	if float64(i) != x {
		panic(fmt.Errorf("can not use %v as int", x))
	}
	return i
}

type funcIf struct{ in Expr }

func (c *funcIf) Eval() interface{}  { return c }
func (c *funcIf) Type() reflect.Type { return funcIf_t }
func (c *funcIf) Float() float64     { return c.in.Eval().(float64) } // implements FuncIf

// converts float64 to func()float64
//type float64ToFunc struct{ in Expr }
//
//func (c *float64ToFunc) Eval() interface{}  { return func() float64 { return c.in.Eval().(float64) } }
//func (c *float64ToFunc) Type() reflect.Type { return func_float64_t }
//
//// converts float64 to func()float64
//type funcToFloat64 struct{ in Expr }
//
//func (c *funcToFloat64) Eval() interface{}  { return (c.in.Eval().(func() float64))() }
//func (c *funcToFloat64) Type() reflect.Type { return float64_t }
//
//type vectorToFunc struct{ in Expr }
//
//func (c *vectorToFunc) Eval() interface{} {
//	return func() [3]float64 { return c.in.Eval().([3]float64) }
//}
//func (c *vectorToFunc) Type() reflect.Type { return func_vector_t }
