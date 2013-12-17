package script

import (
	"go/ast"
	"go/token"
	"io/ioutil"
)

func (w *World) compileSource(n *ast.CallExpr) Expr {
	if len(n.Args) != 1 {
		panic(err(n.Pos(), "source() needs 1 string argument, got", len(n.Args)))
	}
	arg := n.Args[0]
	if lit, ok := arg.(*ast.BasicLit); ok && lit.Kind == token.STRING {

		code, err1 := ioutil.ReadFile(lit.Value[1 : len(lit.Value)-1])
		if err1 != nil {
			panic(err(n.Pos(), err1))
		}
		block, err2 := w.Compile(string(code))
		if err1 != nil {
			panic(err(n.Pos(), err2))
		}
		return block
	} else {
		panic(err(n.Pos(), "source() needs literal string argument"))
	}
}
