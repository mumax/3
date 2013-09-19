package script

import (
	"go/ast"
	"go/token"
	"reflect"
)

// compiles a binary expression x 'op' y
func (w *World) compileBinaryExpr(n *ast.BinaryExpr) Expr {
	x := typeConv(n.Pos(), w.compileExpr(n.X), float64_t)
	y := typeConv(n.Pos(), w.compileExpr(n.Y), float64_t)
	switch n.Op {
	default:
		panic(err(n.Pos(), "not allowed:", n.Op))
	case token.ADD:
		return &add{binaryExpr{x, y}}
	case token.SUB:
		return &sub{binaryExpr{x, y}}
	case token.MUL:
		return &mul{binaryExpr{x, y}}
	case token.QUO:
		return &quo{binaryExpr{x, y}}
	case token.LSS:
		return &lss{comp{x, y}}
	case token.GTR:
		return &gtr{comp{x, y}}
	case token.LEQ:
		return &leq{comp{x, y}}
	case token.GEQ:
		return &geq{comp{x, y}}
	}
}

// abstract superclass for all binary expressions
type binaryExpr struct{ x, y Expr }

func (b *binaryExpr) Type() reflect.Type { return float64_t }
func (b *binaryExpr) Const() bool        { return Const(b.x) && Const(b.y) }

type add struct{ binaryExpr }
type sub struct{ binaryExpr }
type mul struct{ binaryExpr }
type quo struct{ binaryExpr }

func (b *add) Eval() interface{} { return b.x.Eval().(float64) + b.y.Eval().(float64) }
func (b *sub) Eval() interface{} { return b.x.Eval().(float64) - b.y.Eval().(float64) }
func (b *mul) Eval() interface{} { return b.x.Eval().(float64) * b.y.Eval().(float64) }
func (b *quo) Eval() interface{} { return b.x.Eval().(float64) / b.y.Eval().(float64) }

type comp binaryExpr

func (b *comp) Type() reflect.Type { return bool_t }
func (b *comp) Const() bool        { return Const(b.x) && Const(b.y) }

type lss struct{ comp }
type gtr struct{ comp }
type leq struct{ comp }
type geq struct{ comp }

func (b *lss) Eval() interface{} { return b.x.Eval().(float64) < b.y.Eval().(float64) }
func (b *gtr) Eval() interface{} { return b.x.Eval().(float64) > b.y.Eval().(float64) }
func (b *leq) Eval() interface{} { return b.x.Eval().(float64) <= b.y.Eval().(float64) }
func (b *geq) Eval() interface{} { return b.x.Eval().(float64) >= b.y.Eval().(float64) }
