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

	stmt := &forStmt{init: &nop{}, cond: &nop{}, post: &nop{}, body: &nop{}}
	if n.Init != nil {
		stmt.init = w.compileStmt(n.Init)
	}
	if n.Cond != nil {
		stmt.cond = typeConv(n.Cond.Pos(), w.compileExpr(n.Cond), bool_t)
	} else {
		stmt.cond = boolLit(true)
	}
	if n.Post != nil {
		stmt.post = w.compileStmt(n.Post)
	}
	if n.Body != nil {
		stmt.body = w.compileBlockStmt_noScope(n.Body)
	}
	return stmt
}

type nop struct{ void }

func (e *nop) Child() []Expr     { return nil }
func (e *nop) Eval() interface{} { return nil }
func (e *nop) Fix() Expr         { return e }

func (e *forStmt) Child() []Expr {
	return []Expr{e.init, e.cond, e.post, e.body}
}
