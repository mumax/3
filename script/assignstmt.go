package script

import (
	"go/ast"
)

type assignStmt struct {
	lhs lvalue
	rhs expr
}

// compile assign statement
func (w *World) compileAssignStmt(a *ast.AssignStmt) *assignStmt {
	if len(a.Lhs) != 1 || len(a.Rhs) != 1 {
		panic(err("multiple assignment not allowed"))
	}
	lhs, rhs := a.Lhs[0], a.Rhs[0]
	r := w.compileExpr(rhs)
	switch concrete := lhs.(type) {
	default:
		panic(err("cannot assign to", typ(lhs)))
	case *ast.Ident:
		if l, ok := w.resolve(concrete.Name).(lvalue); ok {
			typecheck(l.Type(), r.Type())
			return &assignStmt{l, r}
		} else {
			panic(err("cannot assign to", concrete.Name))
		}
	}
}

func (a *assignStmt) Exec() {
	a.lhs.Set(a.rhs.Eval())
}
