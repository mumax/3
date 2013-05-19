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
		panic(err("no multiple assignment"))
	}
	return w.compile1Assign(a.Lhs[0], a.Rhs[0])
}

// compile single assignment like a = b
func (w *World) compile1Assign(lhs, rhs ast.Expr) *assignStmt {
	switch concrete := lhs.(type) {
	default:
		panic(err("cannot assign to", typ(lhs)))
	case *ast.Ident:
		if l, ok := w.resolve(concrete.Name).(lvalue); ok {
			return &assignStmt{l, w.compileExpr(rhs)}
		} else {
			panic(err("cannot assign to", concrete.Name))
		}
	}
}

func (a *assignStmt) Exec() {
	a.lhs.Set(a.rhs.Eval()[0])
}
