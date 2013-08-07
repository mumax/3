package script

import (
	"log"
	"math"
)

// Loads standard functions into the world.
func (w *World) LoadStdlib() {

	// literals
	w.declare("true", boolLit(true))
	w.declare("false", boolLit(false))

	// io
	w.Func("print", myprint)

	// math
	w.Func("square", square)
	w.Func("abs", math.Abs)
	w.Func("acos", math.Acos)
	w.Func("acosh", math.Acosh)
	w.Func("asin", math.Asin)
	w.Func("asinh", math.Asinh)
	w.Func("atan", math.Atan)
	w.Func("atanh", math.Atanh)
	w.Func("cbrt", math.Cbrt)
	w.Func("ceil", math.Ceil)
	w.Func("cos", math.Cos)
	w.Func("cosh", math.Cosh)
	w.Func("erf", math.Erf)
	w.Func("erfc", math.Erfc)
	w.Func("exp", math.Exp)
	w.Func("exp2", math.Exp2)
	w.Func("expm1", math.Expm1)
	w.Func("floor", math.Floor)
	w.Func("gamma", math.Gamma)
	w.Func("j0", math.J0)
	w.Func("j1", math.J1)
	w.Func("log", math.Log)
	w.Func("log10", math.Log10)
	w.Func("log1p", math.Log1p)
	w.Func("log2", math.Log2)
	w.Func("logb", math.Logb)
	w.Func("sin", math.Sin)
	w.Func("sinh", math.Sinh)
	w.Func("sqrt", math.Sqrt)
	w.Func("tan", math.Tan)
	w.Func("tanh", math.Tanh)
	w.Func("trunc", math.Trunc)
	w.Func("y0", math.Y0)
	w.Func("y1", math.Y1)
	w.Func("ilogb", math.Ilogb)
	w.Func("pow10", math.Pow10)
	w.Func("atan2", math.Atan2)
	w.Func("hypot", math.Hypot)
	w.Func("remainder", math.Remainder)
	w.Func("max", math.Max)
	w.Func("min", math.Min)
	w.Func("mod", math.Mod)
	w.Func("pow", math.Pow)
	w.Func("yn", math.Yn)
	w.Func("jn", math.Jn)
	w.Func("ldexp", math.Ldexp)
	w.Func("isInf", math.IsInf)
	w.Func("isNaN", math.IsNaN)
	w.declare("pi", floatLit(math.Pi))
	w.declare("inf", floatLit(math.Inf(1)))
}

func myprint(msg ...interface{}) {
	log.Println(msg...)
}

func square(x float64) float64 {
	return x * x
}
