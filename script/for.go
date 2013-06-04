package script

import (
	"go/ast"
)

// for statement
type forStmt struct {
	init, cond, post, body Expr
	void
}

func (b *forStmt) Eval() interface{} {
	for b.init.Eval(); b.cond.Eval().(bool); b.post.Eval() {
		b.body.Eval()
	}
	return nil // void
}

func (w *World) compileForStmt(n *ast.ForStmt) *forStmt {
	w.EnterScope()
	defer w.ExitScope()

	return &forStmt{
		init: w.compileStmt(n.Init),
		cond: typeConv(n.Cond.Pos(), w.compileExpr(n.Cond), bool_t),
		post: w.compileStmt(n.Post),
		body: w.compileBlockStmt_noScope(n.Body)}
}
