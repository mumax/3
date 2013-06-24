package script

import (
	"fmt"
	"go/ast"
	"reflect"
	"strings"
	"unicode"
)

type selector struct {
	x      Expr
	method string
}

// compiles a selector statement like x.sel
func (w *World) compileSelectorStmt(n *ast.SelectorExpr) Expr {
	x := w.compileExpr(n.X)
	t := x.Type()

	sel := strings.ToLower(n.Sel.Name)
	N := ""
	for i := 0; i < t.NumMethod(); i++ {
		name := t.Method(i).Name
		if strings.ToLower(name) == sel && unicode.IsUpper(rune(name[0])) {
			N = t.Method(i).Name
			break
		}
	}
	if N == "" {
		panic(err(n.Pos(), t, "has no method", n.Sel.Name))
	}
	return &selector{x, N}
}

func (e *selector) Eval() interface{} {
	obj := reflect.ValueOf(e.x.Eval())
	meth := obj.MethodByName(e.method)
	if meth.Kind() == 0 {
		panic(fmt.Sprint(e.x, " has no method ", e.method))
	}
	return meth.Interface()
}

func (e *selector) Type() reflect.Type {
	return reflect.New(e.x.Type()).Elem().MethodByName(e.method).Type()
}
