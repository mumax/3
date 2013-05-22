package script

import (
	"go/ast"
	"reflect"
)

// an expression can be evaluated
type Expr interface {
	Eval() interface{}
	Type() reflect.Type
}

// compiles an expression
func (w *World) compileExpr(e ast.Expr) Expr {
	switch concrete := e.(type) {
	default:
		panic(err(e.Pos(), "not allowed:", typ(e)))
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
