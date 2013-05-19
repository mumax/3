package script

import (
	"go/ast"
	"go/token"
	"reflect"
)

type binaryExpr struct {
	x, y expr
}

func (b *binaryExpr) Type() reflect.Type {
	return b.x.Type() // assumes it has been typechecked
}

type add struct{ binaryExpr }

func (b *add) Eval() interface{} { return b.x.Eval().(float64) + b.y.Eval().(float64) }

type sub struct{ binaryExpr }

func (b *sub) Eval() interface{} { return b.x.Eval().(float64) - b.y.Eval().(float64) }

type mul struct{ binaryExpr }

func (b *mul) Eval() interface{} { return b.x.Eval().(float64) * b.y.Eval().(float64) }

type quo struct{ binaryExpr }

func (b *quo) Eval() interface{} { return b.x.Eval().(float64) / b.y.Eval().(float64) }

func (w *World) compileBinaryExpr(n *ast.BinaryExpr) expr {
	x := w.compileExpr(n.X)
	y := w.compileExpr(n.Y)
	typecheck(x.Type(), y.Type())
	switch n.Op {
	default:
		panic(err("not allowed:", n.Op))
	case token.ADD:
		return &add{binaryExpr{x, y}}
	case token.SUB:
		return &sub{binaryExpr{x, y}}
	case token.MUL:
		return &mul{binaryExpr{x, y}}
	case token.QUO:
		return &quo{binaryExpr{x, y}}
	}
}
