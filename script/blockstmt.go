package script

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"reflect"
)

// block statement is a list of statements.
type blockStmt struct {
	children []Expr
	node     []ast.Node
}

// does not enter scope because it does not necessarily needs to (e.g. for, if).
func (w *World) compileBlockStmt_noScope(n *ast.BlockStmt) *blockStmt {
	b := &blockStmt{}
	for _, s := range n.List {
		b.append(w.compileStmt(s), s)
	}
	return b
}

func (b *blockStmt) append(s Expr, n ast.Node) {
	b.children = append(b.children, s)
	b.node = append(b.node, n)
}

func (b *blockStmt) Eval() interface{} {
	for _, s := range b.children {
		s.Eval()
	}
	return nil
}

func (b *blockStmt) Type() reflect.Type {
	return nil
}

func (b *blockStmt) Child() []Expr {
	return b.children
}

func (b *blockStmt) Format() string {
	var buf bytes.Buffer
	fset := token.NewFileSet()
	for i := range b.children {
		format.Node(&buf, fset, b.node[i])
		fmt.Fprintln(&buf)
	}
	return buf.String()
}
