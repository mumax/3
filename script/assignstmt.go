package script

import (
	"go/ast"
)

type assignStmt struct {
	lhs assigner
	rhs expr
}

func (w *World) compileAssignStmt(a *ast.AssignStmt) *assignStmt {
	if len(a.Lhs) != 1 || len(a.Rhs) != 1 {
		panic(err("no multiple assignment"))
	}
	return w.compile1Assign(a.Lhs[0], a.Rhs[0])
}

func (w *World) compile1Assign(lhs, rhs ast.Expr) *assignStmt {
	switch concrete := lhs.(type) {
	default:
		panic(err("cannot assign to", typ(lhs)))
	case *ast.Ident:
		if l, ok := w.resolve(concrete.Name).(assigner); ok {
			return &assignStmt{l, rhs}
		} else {
			panic(err("cannot assign to", concrete.Name))
		}
	}
}

type assigner interface {
	Set(func() interface{})
}
