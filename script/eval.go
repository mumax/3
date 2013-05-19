package script

// Eval with panic on error.
func (w *World) MustEval(src string) []interface{} {
	expr := w.MustCompileExpr(src)
	return expr.Eval()
}

// Evaluates src, which must be an expression, and returns the result(s). E.g.:
// 	world.Eval("1+1")      // returns []interface{2}
// 	world.Eval("print(1)") // returns []interface{} (nothing)
func (w *World) Eval(src string) (ret []interface{}, err error) {
	expr, err := w.CompileExpr(src)
	if err != nil {
		return nil, err
	}
	return expr.Eval(), nil
}

// Convenience wrapper for MustEval, converts return value to float64
func (w *World) EvalFloat64(src string) float64 {
	return (w.MustEval(src)[0]).(float64) // TODO: check len
}
