package script

import (
	"go/ast"
	"reflect"
)

type call struct {
	f    function
	args []Expr
}

func (w *World) compileCallExpr(n *ast.CallExpr) Expr {

	// compile function/method
	var f *function
	var receiver Expr
	var fname string
	{
		switch concrete := n.Fun.(type) {
		default:
			panic(err(n.Pos(), "not allowed:", typ(n.Fun)))
		case *ast.Ident: // function call
			if fn, ok := w.resolve(n.Pos(), concrete.Name).(*function); ok {
				f = fn
				fname = concrete.Name
			} else {
				panic(err(n.Pos(), "can not call", concrete.Name))
			}
		case *ast.SelectorExpr: // method call
			receiver = w.compileExpr(concrete.X)
			if val, ok := receiver.Type().MethodByName(concrete.Sel.Name); ok {
				f = &function{val.Func}
				fname = concrete.Sel.Name
			} else {
				panic(err(n.Pos(), typ(receiver.Type()), "does not have method", concrete.Sel.Name))
			}
		}
	}

	args := make([]Expr, len(n.Args))
	variadic := f.Type().IsVariadic()
	for i := range args {
		if variadic {
			args[i] = w.compileExpr(n.Args[i]) // no type check or conversion
		} else {
			args[i] = typeConv(n.Args[i].Pos(), w.compileExpr(n.Args[i]), f.In(i))
		}
	}

	// insert receiver in case of method call
	if receiver != nil {
		args = append([]Expr{receiver}, args...)
	}

	if !variadic && len(n.Args) != f.NumIn() {
		panic(err(n.Pos(), fname, "needs", f.NumIn(), "arguments, got", len(n.Args))) // TODO: varargs
	}

	return &call{*f, args}
}

func (c *call) Eval() interface{} {
	argv := make([]reflect.Value, len(c.args))
	for i := range c.args {
		argv[i] = reflect.ValueOf(c.args[i].Eval())
	}

	ret := c.f.Call(argv)
	assert(len(ret) <= 1)
	if len(ret) == 0 {
		return nil
	} else {
		return ret[0].Interface()
	}
}

func (c *call) Type() reflect.Type {
	return c.f.ReturnType()
}
