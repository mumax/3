package script

import (
	"go/ast"
	"reflect"
)

// block statement is a list of statements.
type blockStmt []Expr

// does not enter scope because it does not necessarily needs to (e.g. for, if).
func (w *World) compileBlockStmt_noScope(n *ast.BlockStmt) *blockStmt {
	B := make(blockStmt, 0, len(n.List))
	b := &B
	for _, s := range n.List {
		b.append(w.compileStmt(s))
	}
	return b
}

func (b *blockStmt) append(s Expr) {
	(*b) = append(*b, s)
}

func (b *blockStmt) Eval() interface{} {
	for _, s := range *b {
		s.Eval()
	}
	return nil
}

func (b *blockStmt) Type() reflect.Type {
	return nil
}
