package script

import (
	"fmt"
	"go/token"
	"reflect"

	"github.com/mumax/3/data"
)

// converts in to an expression of type OutT.
// also serves as type check (not convertible == type error)
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

	case outT == float64_t && inT.AssignableTo(ScalarIf_t):
		return &getScalar{in.Eval().(ScalarIf)}
	case outT == float64_t && inT.AssignableTo(VectorIf_t):
		return &getVector{in.Eval().(VectorIf)}

	// magical expression -> function conversions
	case inT == float64_t && outT.AssignableTo(ScalarFunction_t):
		return &scalFn{in}
	case inT == int_t && outT.AssignableTo(ScalarFunction_t):
		return &scalFn{&intToFloat64{in}}
	case inT == vector_t && outT.AssignableTo(VectorFunction_t):
		return &vecFn{in}
	case inT == bool_t && outT == func_bool_t:
		return &boolToFunc{in}
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
	float64_t        = reflect.TypeFor[float64]()
	bool_t           = reflect.TypeFor[bool]()
	func_float64_t   = reflect.TypeFor[func() float64]()
	func_bool_t      = reflect.TypeFor[func() bool]()
	int_t            = reflect.TypeFor[int]()
	string_t         = reflect.TypeFor[string]()
	vector_t         = reflect.TypeFor[data.Vector]()
	func_vector_t    = reflect.TypeFor[func() data.Vector]()
	ScalarFunction_t = reflect.TypeFor[func(ScalarFunction)]().In(0)
	VectorFunction_t = reflect.TypeFor[func(VectorFunction)]().In(0)
	ScalarIf_t       = reflect.TypeFor[func(ScalarIf)]().In(0)
	VectorIf_t       = reflect.TypeFor[func(VectorIf)]().In(0)
)

// converts int to float64
type intToFloat64 struct{ in Expr }

func (c *intToFloat64) Eval() any          { return float64(c.in.Eval().(int)) }
func (c *intToFloat64) Type() reflect.Type { return float64_t }
func (c *intToFloat64) Child() []Expr      { return []Expr{c.in} }
func (c *intToFloat64) Fix() Expr          { return &intToFloat64{in: c.in.Fix()} }

// converts float64 to int
type float64ToInt struct{ in Expr }

func (c *float64ToInt) Eval() any          { return safe_int(c.in.Eval().(float64)) }
func (c *float64ToInt) Type() reflect.Type { return int_t }
func (c *float64ToInt) Child() []Expr      { return []Expr{c.in} }
func (c *float64ToInt) Fix() Expr          { return &float64ToInt{in: c.in.Fix()} }

type boolToFunc struct{ in Expr }

func (c *boolToFunc) Eval() any          { return func() bool { return c.in.Eval().(bool) } }
func (c *boolToFunc) Type() reflect.Type { return func_bool_t }
func (c *boolToFunc) Child() []Expr      { return []Expr{c.in} }
func (c *boolToFunc) Fix() Expr          { return &boolToFunc{in: c.in.Fix()} }

type getScalar struct{ in ScalarIf }
type getVector struct{ in VectorIf }

func (c *getScalar) Eval() any          { return c.in.Get() }
func (c *getScalar) Type() reflect.Type { return float64_t }
func (c *getScalar) Child() []Expr      { return nil }
func (c *getScalar) Fix() Expr          { return NewConst(c) }

func (c *getVector) Eval() any          { return c.in.Get() }
func (c *getVector) Type() reflect.Type { return vector_t }
func (c *getVector) Child() []Expr      { return nil }
func (c *getVector) Fix() Expr          { return NewConst(c) }

func safe_int(x float64) int {
	i := int(x)
	if float64(i) != x {
		panic(fmt.Errorf("can not use %v as int", x))
	}
	return i
}

type ScalarIf interface {
	Get() float64
} // TODO: Scalar

type VectorIf interface {
	Get() data.Vector
} // TODO: Vector
