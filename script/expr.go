package script

import (
	"go/ast"
	"reflect"
)

// an expression can be evaluated
type expr interface {
	Eval() interface{}
	Type() reflect.Type
}

// compiles an expression
func (w *World) compileExpr(e ast.Expr) expr {
	switch concrete := e.(type) {
	default:
		panic(err("not allowed:", typ(e)))
	case *ast.Ident:
		return w.resolve(concrete.Name)
	case *ast.BasicLit:
		return w.compileBasicLit(concrete)
	case *ast.BinaryExpr:
		return w.compileBinaryExpr(concrete)
	case *ast.CallExpr:
		return w.compileCallExpr(concrete)
	case *ast.ParenExpr:
		return w.compileExpr(concrete.X)
	}
}
