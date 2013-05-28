package script

import (
	"go/ast"
	"reflect"
)

// compiles expression or statement
func (w *World) compile(n ast.Node) Expr {
	switch concrete := n.(type) {
	case ast.Stmt:
		return w.compileStmt(concrete)
	case ast.Expr:
		return w.compileExpr(concrete)
	default:
		panic(err(n.Pos(), "not allowed"))
	}
}

// compiles a statement
func (w *World) compileStmt(st ast.Stmt) Expr {
	switch concrete := st.(type) {
	default:
		panic(err(st.Pos(), "not allowed:", typ(st)))
	case *ast.AssignStmt:
		return w.compileAssignStmt(concrete)
	case *ast.ExprStmt:
		return w.compileExpr(concrete.X)
	}
}

// embed to get Type() that returns nil
type void struct{}

func (v *void) Type() reflect.Type {
	return nil
}
