package script

import "reflect"

func typeconv(in Expr, outT reflect.Type) Expr {
	inT := in.Type()
	switch {
	default:
		panic(err("type mismatch: can not use type ", inT, " as ", outT))
	case inT == outT:
		return in
	case outT == float64_t && inT == int_t:
		return &intToFloat64{in}
	}
}

var (
	float64_t = reflect.TypeOf(float64(0))
	int_t     = reflect.TypeOf(int(0))
)

type intToFloat64 struct {
	in Expr
}

func (c *intToFloat64) Eval() interface{} {
	return float64(c.in.Eval().(int))
}

func (c *intToFloat64) Type() reflect.Type { return float64_t }
