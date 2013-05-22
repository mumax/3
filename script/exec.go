package script

// Eval with panic on error.
func (w *World) MustEval(src string) interface{} {
	Expr := w.MustCompileExpr(src)
	return Expr.Eval()
}

// Eval compiles and evaluates src, which must be an expression, and returns the result(s). E.g.:
// 	world.Eval("1+1")      // returns 2, nil
func (w *World) Eval(src string) (ret interface{}, err error) {
	Expr, err := w.CompileExpr(src)
	if err != nil {
		return nil, err
	}
	return Expr.Eval(), nil
}
