package script

import (
	"go/ast"
	"go/token"
	"reflect"
)

// compiles a (single) assign statement lhs = rhs
func (w *World) compileAssignStmt(a *ast.AssignStmt) Expr {
	if len(a.Lhs) != 1 || len(a.Rhs) != 1 {
		panic(err(a.Pos(), "multiple assignment not allowed"))
	}
	lhs, rhs := a.Lhs[0], a.Rhs[0]
	r := w.compileExpr(rhs)

	switch a.Tok {
	default:
		panic(err(a.Pos(), a.Tok, "not allowed"))
	case token.ASSIGN: // =
		return w.compileAssign(a, lhs, r)
	case token.DEFINE: // :=
		return w.compileDefine(a, lhs, r)
	case token.ADD_ASSIGN: // +=
		return w.compileAddAssign(a, lhs, r)
	case token.SUB_ASSIGN: // -=
		return w.compileSubAssign(a, lhs, r)
	}
}

// compile a = b
func (w *World) compileAssign(a *ast.AssignStmt, lhs ast.Expr, r Expr) Expr {
	l := w.compileLvalue(lhs)
	return &assignStmt{lhs: l, rhs: typeConv(a.Pos(), r, inputType(l))}
}

// compile a := b
func (w *World) compileDefine(a *ast.AssignStmt, lhs ast.Expr, r Expr) Expr {
	ident, ok := lhs.(*ast.Ident)
	if !ok {
		panic(err(a.Pos(), "non-name on left side of :="))
	}
	addr := reflect.New(r.Type())
	ok = w.safeDeclare(ident.Name, &reflectLvalue{addr.Elem()})
	if !ok {
		panic(err(a.Pos(), "already defined: "+ident.Name))
	}
	return w.compileAssign(a, lhs, r)
}

type assignStmt struct {
	lhs LValue
	rhs Expr
	void
}

func (a *assignStmt) Eval() interface{} {
	a.lhs.SetValue(a.rhs.Eval())
	return nil
}

func (a *assignStmt) Child() []Expr {
	return []Expr{a.lhs, a.rhs}
}

func (w *World) compileAddAssign(a *ast.AssignStmt, lhs ast.Expr, r Expr) Expr {
	l := w.compileLvalue(lhs)
	x := typeConv(a.Pos(), l, float64_t)
	y := typeConv(a.Pos(), r, float64_t)
	sum := &add{binaryExpr{x, y}}
	return &assignStmt{lhs: l, rhs: typeConv(a.Pos(), sum, inputType(l))}
}

func (w *World) compileSubAssign(a *ast.AssignStmt, lhs ast.Expr, r Expr) Expr {
	l := w.compileLvalue(lhs)
	x := typeConv(a.Pos(), l, float64_t)
	y := typeConv(a.Pos(), r, float64_t)
	sub := &sub{binaryExpr{x, y}}
	return &assignStmt{lhs: l, rhs: typeConv(a.Pos(), sub, inputType(l))}
}
