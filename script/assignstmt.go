package script

import "go/ast"

// compiles a (single) assign statement lhs = rhs
func (w *World) compileAssignStmt(a *ast.AssignStmt) Expr {
	if len(a.Lhs) != 1 || len(a.Rhs) != 1 {
		panic(err(a.Pos(), "multiple assignment not allowed"))
	}
	lhs, rhs := a.Lhs[0], a.Rhs[0]
	r := w.compileExpr(rhs)
	switch concrete := lhs.(type) {
	default:
		panic(err(a.Pos(), "cannot assign to", typ(lhs)))
	case *ast.Ident:
		if l, ok := w.resolve(concrete.Name).(lvalue); ok {
			return &assignStmt{lhs: l, rhs: typeconv(r, l.Type())}
		} else {
			panic(err(a.Pos(), "cannot assign to", concrete.Name))
		}
	}
}

type assignStmt struct {
	lhs lvalue
	rhs Expr
	void
}

func (a *assignStmt) Eval() interface{} {
	a.lhs.Set(a.rhs.Eval())
	return nil
}
