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
	case token.LAND:
		return &and{w.newBoolOp(n)}
	case token.LOR:
		return &or{w.newBoolOp(n)}
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

func (b *add) Fix() Expr { return &add{binaryExpr{x: b.x.Fix(), y: b.y.Fix()}} }
func (b *sub) Fix() Expr { return &sub{binaryExpr{x: b.x.Fix(), y: b.y.Fix()}} }
func (b *mul) Fix() Expr { return &mul{binaryExpr{x: b.x.Fix(), y: b.y.Fix()}} }
func (b *quo) Fix() Expr { return &quo{binaryExpr{x: b.x.Fix(), y: b.y.Fix()}} }

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

func (b *lss) Fix() Expr { return &lss{comp{x: b.x.Fix(), y: b.y.Fix()}} }
func (b *gtr) Fix() Expr { return &gtr{comp{x: b.x.Fix(), y: b.y.Fix()}} }
func (b *leq) Fix() Expr { return &leq{comp{x: b.x.Fix(), y: b.y.Fix()}} }
func (b *geq) Fix() Expr { return &geq{comp{x: b.x.Fix(), y: b.y.Fix()}} }
func (b *eql) Fix() Expr { return &eql{comp{x: b.x.Fix(), y: b.y.Fix()}} }
func (b *neq) Fix() Expr { return &neq{comp{x: b.x.Fix(), y: b.y.Fix()}} }

type boolOp struct{ x, y Expr }

func (w *World) newBoolOp(n *ast.BinaryExpr) boolOp {
	x := typeConv(n.Pos(), w.compileExpr(n.X), bool_t)
	y := typeConv(n.Pos(), w.compileExpr(n.Y), bool_t)
	return boolOp{x, y}
}

func (b *boolOp) Child() []Expr      { return []Expr{b.x, b.y} }
func (b *boolOp) Type() reflect.Type { return bool_t }

type and struct{ boolOp }
type or struct{ boolOp }

func (b *and) Eval() interface{} { return b.x.Eval().(bool) && b.y.Eval().(bool) }
func (b *or) Eval() interface{}  { return b.x.Eval().(bool) || b.y.Eval().(bool) }

func (b *and) Fix() Expr { return &and{boolOp{x: b.x.Fix(), y: b.y.Fix()}} }
func (b *or) Fix() Expr  { return &or{boolOp{x: b.x.Fix(), y: b.y.Fix()}} }
