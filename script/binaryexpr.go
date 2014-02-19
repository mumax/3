package script

import (
	"go/ast"
	"go/token"
	"reflect"
)

// compiles a binary expression x 'op' y
func (w *World) compileBinaryExpr(n *ast.BinaryExpr) Expr {
	switch n.Op {
	default:
		panic(err(n.Pos(), "not allowed:", n.Op))
	case token.ADD:
		return &add{w.newBinExpr(n)}
	case token.SUB:
		return &sub{w.newBinExpr(n)}
	case token.MUL:
		return &mul{w.newBinExpr(n)}
	case token.QUO:
		return &quo{w.newBinExpr(n)}
	case token.LSS:
		return &lss{w.newComp(n)}
	case token.GTR:
		return &gtr{w.newComp(n)}
	case token.LEQ:
		return &leq{w.newComp(n)}
	case token.GEQ:
		return &geq{w.newComp(n)}
	case token.EQL:
		return &eql{w.newComp(n)}
	case token.NEQ:
		return &neq{w.newComp(n)}
	}
}

// abstract superclass for all binary expressions
type binaryExpr struct{ x, y Expr }

func (w *World) newBinExpr(n *ast.BinaryExpr) binaryExpr {
	x := typeConv(n.Pos(), w.compileExpr(n.X), float64_t)
	y := typeConv(n.Pos(), w.compileExpr(n.Y), float64_t)
	return binaryExpr{x, y}
}

func (b *binaryExpr) Type() reflect.Type { return float64_t }
func (b *binaryExpr) Child() []Expr      { return []Expr{b.x, b.y} }

type add struct{ binaryExpr }
type sub struct{ binaryExpr }
type mul struct{ binaryExpr }
type quo struct{ binaryExpr }

func (b *add) Eval() interface{} { return b.x.Eval().(float64) + b.y.Eval().(float64) }
func (b *sub) Eval() interface{} { return b.x.Eval().(float64) - b.y.Eval().(float64) }
func (b *mul) Eval() interface{} { return b.x.Eval().(float64) * b.y.Eval().(float64) }
func (b *quo) Eval() interface{} { return b.x.Eval().(float64) / b.y.Eval().(float64) }

type comp binaryExpr

func (w *World) newComp(n *ast.BinaryExpr) comp {
	return comp(w.newBinExpr(n))
}

func (b *comp) Type() reflect.Type { return bool_t }
func (b *comp) Child() []Expr      { return []Expr{b.x, b.y} }

type lss struct{ comp }
type gtr struct{ comp }
type leq struct{ comp }
type geq struct{ comp }
type eql struct{ comp }
type neq struct{ comp }

func (b *lss) Eval() interface{} { return b.x.Eval().(float64) < b.y.Eval().(float64) }
func (b *gtr) Eval() interface{} { return b.x.Eval().(float64) > b.y.Eval().(float64) }
func (b *leq) Eval() interface{} { return b.x.Eval().(float64) <= b.y.Eval().(float64) }
func (b *geq) Eval() interface{} { return b.x.Eval().(float64) >= b.y.Eval().(float64) }
func (b *eql) Eval() interface{} { return b.x.Eval().(float64) == b.y.Eval().(float64) }
func (b *neq) Eval() interface{} { return b.x.Eval().(float64) != b.y.Eval().(float64) }
