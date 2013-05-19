package script

// Eval with panic on error.
func (w *World) MustEval(src string) []interface{} {
	expr := w.MustCompileExpr(src)
	return expr.Eval()
}

// Evaluates src, which must be an expression, and returns the result. E.g.:
// 	world.Eval("1+1") // returns []interface{2}
func (w *World) Eval(src string) (ret []interface{}, err error) {
	expr, err := w.CompileExpr(src)
	if err != nil {
		return nil, err
	}
	return expr.Eval(), nil
}
