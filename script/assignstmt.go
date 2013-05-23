package script

import (
	"go/ast"
	"go/token"
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
	case token.ASSIGN:
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
}

type assignStmt struct {
	lhs LValue
	rhs Expr
	void
}

func (a *assignStmt) Eval() interface{} {
	a.lhs.Set(a.rhs.Eval())
	return nil
}
