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
	case *ast.Ident:
		return w.resolve(concrete.Name)
	case *ast.BasicLit:
		return w.compileBasicLit(concrete)
	}
}
