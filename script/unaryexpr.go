package script

import (
	"go/ast"
	"go/token"
	"reflect"
)

func (w *World) compileUnaryExpr(n *ast.UnaryExpr) Expr {
	x := w.compileExpr(n.X)
	switch n.Op {
	default:
		panic(err(n.Pos(), "not allowed:", n.Op))
	case token.SUB:
		return &minus{typeConv(n.X.Pos(), x, float64_t)}
	}
}

type minus struct{ x Expr }

func (m *minus) Type() reflect.Type { return float64_t }
func (m *minus) Eval() interface{}  { return -m.x.Eval().(float64) }
func (m *minus) Const() bool        { return Const(m.x) }
