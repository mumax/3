package script

import (
	"go/ast"
	"reflect"
)

type vector struct {
	comp [3]Expr
}

func (w *World) compileVector(n *ast.CallExpr) Expr {
	a := n.Args
	if len(a) != 3 {
		panic(err(n.Pos(), "vector needs 3 arguments, got", len(n.Args)))
	}
	v := new(vector)
	for i := range v.comp {
		v.comp[i] = typeConv(n.Args[i].Pos(), w.compileExpr(n.Args[i]), float64_t)
	}
	return v
}

func (v *vector) Eval() interface{} {
	return [3]float64{v.comp[0].Eval().(float64), v.comp[1].Eval().(float64), v.comp[2].Eval().(float64)}
}

func (v *vector) Type() reflect.Type {
	return vector_t
}
