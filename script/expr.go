package script

import (
	"go/ast"
	"reflect"
)

// an expression can be evaluated
type Expr interface {
	Eval() interface{}  // evaluate and return result (nil for void)
	Type() reflect.Type // return type, nil for void
}

// compiles an expression
func (w *World) compileExpr(e ast.Expr) Expr {
	switch concrete := e.(type) {
	default:
		panic(err(e.Pos(), "not allowed:", typ(e)))
	case *ast.Ident:
		return w.resolve(e.Pos(), concrete.Name)
	case *ast.BasicLit:
		return w.compileBasicLit(concrete)
	case *ast.BinaryExpr:
		return w.compileBinaryExpr(concrete)
	case *ast.UnaryExpr:
		return w.compileUnaryExpr(concrete)
	case *ast.CallExpr:
		return w.compileCallExpr(concrete)
	case *ast.ParenExpr:
		return w.compileExpr(concrete.X)
	case *ast.IndexExpr:
		return w.compileIndexExpr(concrete)
	}
	panic(0) // silence gccgo
}
