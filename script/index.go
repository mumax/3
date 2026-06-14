package script

import (
	"go/ast"
	"reflect"
)

func (w *World) compileIndexExpr(n *ast.IndexExpr) Expr {
	x := w.compileExpr(n.X)
	kind := x.Type().Kind()
	if !(kind == reflect.Array || kind == reflect.Slice) {
		panic(err(n.Pos(), "can not index", x.Type()))
	}
	i := typeConv(n.Index.Pos(), w.compileExpr(n.Index), int_t)
	return &index{x, i}
}

type index struct {
	x, index Expr
}

func (e *index) Type() reflect.Type {
	return e.x.Type().Elem()
}
func (e *index) Eval() interface{} {
	x := reflect.ValueOf(e.x.Eval())
	i := e.index.Eval().(int)
	return x.Index(i).Interface()
}

func (e *index) Child() []Expr {
	return []Expr{e.x, e.index}
}

func (e *index) Fix() Expr {
	return &index{x: e.x.Fix(), index: e.index.Fix()}
}
