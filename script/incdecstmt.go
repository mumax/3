package script

import (
	"go/ast"
	"go/token"
	"reflect"
)

func (w *World) compileIncDecStmt(n *ast.IncDecStmt) Expr {
	l := w.compileLvalue(n.X)
	switch n.Tok {
	case token.INC:
		rhs_plus1 := &addone{incdec{typeConv(n.Pos(), l, float64_t)}}
		return &assignStmt{lhs: l, rhs: typeConv(n.Pos(), rhs_plus1, l.Type())}
	case token.DEC:
		rhs_minus1 := &subone{incdec{typeConv(n.Pos(), l, float64_t)}}
		return &assignStmt{lhs: l, rhs: typeConv(n.Pos(), rhs_minus1, l.Type())}
	default:
		panic(err(n.Pos(), "not allowed:", n.Tok))
	}
}

type incdec struct{ x Expr }

func (e *incdec) Type() reflect.Type { return float64_t }
func (e *incdec) Child() []Expr      { return []Expr{e.x} }
func (e *incdec) Fix() Expr          { panic(invalid_closure) }

type addone struct{ incdec }
type subone struct{ incdec }

func (s *addone) Eval() interface{} { return s.x.Eval().(float64) + 1 }
func (s *subone) Eval() interface{} { return s.x.Eval().(float64) - 1 }
