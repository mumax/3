package script

import (
	"go/ast"
	"go/token"
	"log"
	"reflect"
)

// compiles a (single) assign statement lhs = rhs
func (w *World) compileAssignStmt(a *ast.AssignStmt) Expr {
	if len(a.Lhs) != 1 || len(a.Rhs) != 1 {
		panic(err(a.Pos(), "multiple assignment not allowed"))
	}
	lhs, rhs := a.Lhs[0], a.Rhs[0]
	r := w.compileExpr(rhs)

	switch a.Tok {
	default:
		panic(err(a.Pos(), a.Tok, "not allowed"))
	case token.ASSIGN: // =
		return w.compileAssign(a, lhs, r)
	case token.DEFINE: // :=
		return w.compileDefine(a, lhs, r)
	}

}

// compile a = b
func (w *World) compileAssign(a *ast.AssignStmt, lhs ast.Expr, r Expr) Expr {
	switch concrete := lhs.(type) {
	default:
		panic(err(a.Pos(), "cannot assign to", typ(lhs)))
	case *ast.Ident:
		if l, ok := w.resolve(a.Pos(), concrete.Name).(LValue); ok {
			return &assignStmt{lhs: l, rhs: typeConv(a.Pos(), r, inputType(l))}
		} else {
			panic(err(a.Pos(), "cannot assign to", concrete.Name))
		}
	}
}

// compile a := b
func (w *World) compileDefine(a *ast.AssignStmt, lhs ast.Expr, r Expr) Expr {
	// TODO: catch
	ident, ok := lhs.(*ast.Ident)
	if !ok {
		panic(err(a.Pos(), "non-name on left side of :="))
	}
	addr := reflect.New(r.Type())
	w.declare(ident.Name, &reflectLvalue{reflectROnly{addr.Elem()}})
	log.Println("declare", ident.Name)
	return w.compileAssign(a, lhs, r)
}

type assignStmt struct {
	lhs LValue
	rhs Expr
	void
}

func (a *assignStmt) Eval() interface{} {
	a.lhs.SetValue(a.rhs.Eval())
	return nil
}
