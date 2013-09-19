package script

import (
	"go/ast"
	"reflect"
)

type call struct {
	f    Expr
	args []Expr
}

func (w *World) compileCallExpr(n *ast.CallExpr) Expr {
	// compile function or method to be called
	var f Expr
	var fname string
	switch Fun := n.Fun.(type) {
	default:
		panic(err(n.Pos(), "not allowed:", typ(n.Fun)))
	case *ast.Ident: // function call
		f = w.compileExpr(Fun)
		fname = Fun.Name
	case *ast.SelectorExpr: // method call
		f = w.compileSelectorStmt(Fun)
		fname = Fun.Sel.Name
	}
	if f.Type().Kind() != reflect.Func {
		panic(err(n.Pos(), "can not call", n))
	}

	// compile and check args
	args := make([]Expr, len(n.Args))
	variadic := f.Type().IsVariadic()
	if !variadic && len(n.Args) != f.Type().NumIn() {
		panic(err(n.Pos(), fname, "needs", f.Type().NumIn(), "arguments, got", len(n.Args))) // TODO: varargs
	}
	for i := range args {
		if variadic {
			args[i] = w.compileExpr(n.Args[i]) // no type check or conversion
		} else {
			args[i] = typeConv(n.Args[i].Pos(), w.compileExpr(n.Args[i]), f.Type().In(i))
		}
	}
	return &call{f, args}
}

func (c *call) Eval() interface{} {
	// evaluate and pack arguments
	argv := make([]reflect.Value, len(c.args))
	for i := range c.args {
		argv[i] = reflect.ValueOf(c.args[i].Eval())
	}

	// evaluate function
	f := reflect.ValueOf(c.f.Eval())

	// call
	ret := f.Call(argv)

	// at most 1 return value allowed
	assert(len(ret) <= 1)
	if len(ret) == 0 {
		return nil
	} else {
		return ret[0].Interface()
	}
}

// return type of call
func (c *call) Type() reflect.Type {
	switch c.f.Type().NumOut() {
	case 0:
		return nil // "void"
	case 1:
		return c.f.Type().Out(0)
	default:
		panic("bug: multiple return values not allowed")
	}
}
