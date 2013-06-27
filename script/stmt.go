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
	panic(0) // silence gccgo
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
	case *ast.ForStmt:
		return w.compileForStmt(concrete)
	case *ast.IncDecStmt:
		return w.compileIncDecStmt(concrete)
	case *ast.BlockStmt:
		w.EnterScope()
		defer w.ExitScope()
		return w.compileBlockStmt_noScope(concrete)
	}
	panic(0) // silence gccgo
}

// embed to get Type() that returns nil
type void struct{}

func (v *void) Type() reflect.Type {
	return nil
}
