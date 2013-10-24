package script

import (
	"go/ast"
)

// if statement
type ifStmt struct {
	cond, body Expr
	void
}

func (b *ifStmt) Eval() interface{} {
	if b.cond.Eval().(bool) {
		b.body.Eval()
	}
	return nil // void
}

func (w *World) compileIfStmt(n *ast.IfStmt) *ifStmt {
	w.EnterScope()
	defer w.ExitScope()

	return &ifStmt{
		cond: typeConv(n.Cond.Pos(), w.compileExpr(n.Cond), bool_t),
		body: w.compileBlockStmt_noScope(n.Body)}
}

func (e *ifStmt) Child() []Expr {
	return []Expr{e.cond, e.body}
}
