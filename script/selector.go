package script

import (
	"go/ast"
	"reflect"
	"strings"
)

type selector struct {
	x      Expr
	method int
}

func (w *World) compileSelectorStmt(n *ast.SelectorExpr) Expr {
	x := w.compileExpr(n.X)
	t := x.Type()

	sel := strings.ToLower(n.Sel.Name)
	N := -1
	for i := 0; i < t.NumMethod(); i++ {
		if strings.ToLower(t.Method(i).Name) == sel {
			N = i
			break
		}
	}
	return &selector{x, N}
}

func (e *selector) Eval() interface{} {
	return reflect.ValueOf(e.x.Eval()).Method(e.method).Interface()
}

func (e *selector) Type() reflect.Type {
	return e.x.Type().Method(e.method).Type
}
