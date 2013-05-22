package script

import (
	"go/ast"
	"reflect"
)

// compiles a statement
func (w *World) compileStmt(st ast.Stmt) Expr {
	switch concrete := st.(type) {
	default:
		panic(err("not allowed:", typ(st)))
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
