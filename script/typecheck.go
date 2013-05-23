package script

import (
	"go/token"
	"reflect"
)

func typeConv(pos token.Pos, in Expr, outT reflect.Type) Expr {
	inT := in.Type()
	switch {
	default:
		panic(err(pos, "type mismatch: can not use type", inT, "as", outT)) // TODO: add pos!
	case inT == outT:
		return in
	case outT == float64_t && inT == int_t:
		return &intToFloat64{in}
	}
}

var (
	float64_t = reflect.TypeOf(float64(0))
	int_t     = reflect.TypeOf(int(0))
	string_t  = reflect.TypeOf("")
)

type intToFloat64 struct {
	in Expr
}

func (c *intToFloat64) Eval() interface{} {
	return float64(c.in.Eval().(int))
}

func (c *intToFloat64) Type() reflect.Type { return float64_t }

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
