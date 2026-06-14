package script

import (
	"go/ast"
)

// if statement
type ifStmt struct {
	cond, body, else_ Expr
	void
}

func (b *ifStmt) Eval() interface{} {
	if b.cond.Eval().(bool) {
		b.body.Eval()
	} else {
		if b.else_ != nil {
			b.else_.Eval()
		}
	}
	return nil // void
}

func (w *World) compileIfStmt(n *ast.IfStmt) *ifStmt {
	w.EnterScope()
	defer w.ExitScope()

	stmt := &ifStmt{
		cond: typeConv(n.Cond.Pos(), w.compileExpr(n.Cond), bool_t),
		body: w.compileBlockStmt_noScope(n.Body)}
	if n.Else != nil {
		stmt.else_ = w.compileStmt(n.Else)
	}

	return stmt
}

func (e *ifStmt) Child() []Expr {
	child := []Expr{e.cond, e.body, e.else_}
	if e.else_ == nil {
		child = child[:2]
	}
	return child
}
