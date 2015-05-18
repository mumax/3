package script

import (
	"go/ast"
	"reflect"
)

// an expression can be evaluated
type Expr interface {
	Eval() interface{}  // evaluate and return result (nil for void)
	Type() reflect.Type // return type, nil for void
	Child() []Expr
	Fix() Expr // replace all variables by their current value, except for the time "t".
}

// compiles an expression
func (w *World) compileExpr(e ast.Expr) Expr {
	switch e := e.(type) {
	default:
		panic(err(e.Pos(), "not allowed:", typ(e)))
	case *ast.Ident:
		return w.resolve(e.Pos(), e.Name)
	case *ast.BasicLit:
		return w.compileBasicLit(e)
	case *ast.BinaryExpr:
		return w.compileBinaryExpr(e)
	case *ast.UnaryExpr:
		return w.compileUnaryExpr(e)
	case *ast.CallExpr:
		return w.compileCallExpr(e)
	case *ast.ParenExpr:
		return w.compileExpr(e.X)
	case *ast.IndexExpr:
		return w.compileIndexExpr(e)
	}
}
