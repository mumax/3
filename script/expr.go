package script

import (
	"go/ast"
)

type expr interface {
	Eval() []interface{}
	NumOut() int
}

func (w *World) compileExpr(e ast.Expr) expr {
	switch concrete := e.(type) {
	default:
		panic(err("not allowed:", typ(e)))
	case *ast.BasicLit:
		return w.compileBasicLit(concrete)
	}
}
