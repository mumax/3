package script

import (
	"go/ast"
	"reflect"
)

// compiles expression or statement
func (w *World) compile(n ast.Node) Expr {
	switch n := n.(type) {
	case ast.Stmt:
		return w.compileStmt(n)
	case ast.Expr:
		return w.compileExpr(n)
	default:
		panic(err(n.Pos(), "not allowed"))
	}
}

// compiles a statement
func (w *World) compileStmt(st ast.Stmt) Expr {
	switch st := st.(type) {
	default:
		panic(err(st.Pos(), "not allowed:", typ(st)))
	case *ast.AssignStmt:
		return w.compileAssignStmt(st)
	case *ast.ExprStmt:
		return w.compileExpr(st.X)
	case *ast.ForStmt:
		return w.compileForStmt(st)
	case *ast.IncDecStmt:
		return w.compileIncDecStmt(st)
	case *ast.BlockStmt:
		w.EnterScope()
		defer w.ExitScope()
		return w.compileBlockStmt_noScope(st)
	}
}

// embed to get Type() that returns nil
type void struct{}

func (v *void) Type() reflect.Type { return nil }
func (v *void) Const() bool        { return false }
