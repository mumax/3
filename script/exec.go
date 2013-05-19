package script

// Exec compiles and executes the source statements.
func (w *World) Exec(src string) error {
	code, err := w.Compile(src)
	if err != nil {
		return err
	}
	code.Exec()
	return nil
}

// Exec with panic on error.
func (w *World) MustExec(src string) {
	code := w.MustCompile(src)
	code.Exec()
}

// Eval with panic on error.
func (w *World) MustEval(src string) interface{} {
	expr := w.MustCompileExpr(src)
	return expr.Eval()
}

// Eval compiles and evaluates src, which must be an expression, and returns the result(s). E.g.:
// 	world.Eval("1+1")      // returns 2, nil
func (w *World) Eval(src string) (ret interface{}, err error) {
	expr, err := w.CompileExpr(src)
	if err != nil {
		return nil, err
	}
	return expr.Eval(), nil
}
