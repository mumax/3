package script

import (
	"go/ast"
	"reflect"
)

type call interface {
	Expr
}

func (w *World) compileCallExpr(n *ast.CallExpr) call {
	// only call idents for now
	id, ok := (n.Fun).(*ast.Ident)
	if !ok {
		panic(err(n.Pos(), "can not call", typ(n.Fun)))
	}
	// only call reflectFunc for now
	f, ok2 := w.resolve(n.Pos(), id.Name).(*reflectFunc)
	if !ok2 {
		panic(err(n.Pos(), "can not call", id.Name))
	}
	// check args count. no strict check for varargs
	variadic := f.fn.Type().IsVariadic()
	if !variadic && len(n.Args) != f.NumIn() {
		panic(err(n.Pos(), id.Name, "needs", f.NumIn(), "arguments, got", len(n.Args))) // TODO: varargs
	}
	// convert args
	args := make([]Expr, len(n.Args))
	for i := range args {
		if variadic {
			args[i] = w.compileExpr(n.Args[i]) // no type check or conversion
		} else {
			args[i] = typeconv(n.Args[i].Pos(), w.compileExpr(n.Args[i]), f.In(i))
		}
	}
	return &reflectCall{f, args}
}

type reflectCall struct {
	f    *reflectFunc
	args []Expr
}

func (c *reflectCall) Eval() interface{} {
	argv := make([]reflect.Value, len(c.args))
	for i := range c.args {
		argv[i] = reflect.ValueOf(c.args[i].Eval())
	}
	ret := c.f.fn.Call(argv)
	assert(len(ret) <= 1)
	if len(ret) == 0 {
		return nil
	} else {
		return ret[0].Interface()
	}
}

// TODO: if arg expr is of type EvalValue()reflect.Value, don't convert to interface{} and back to reflect.Value

func (c *reflectCall) Exec() {
	c.Eval()
}

func (c *reflectCall) Type() reflect.Type {
	return c.f.Type()
}
