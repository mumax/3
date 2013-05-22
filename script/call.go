package script

import (
	"go/ast"
	"reflect"
)

type call interface {
	Expr
	Stmt
}

// TODO: might become func.curry, so it can be optimized
func (w *World) compileCallExpr(n *ast.CallExpr) call {
	id, ok := (n.Fun).(*ast.Ident)
	if !ok {
		panic(err("can not call", typ(n.Fun)))
	}
	f, ok2 := w.resolve(id.Name).(*reflectFunc)
	if !ok2 {
		panic(err("can not call", id.Name))
	}
	if len(n.Args) != f.NumIn() {
		panic(err(id.Name, "needs", f.NumIn(), "arguments, got", len(n.Args))) // TODO: varargs
	}
	args := make([]Expr, len(n.Args))
	for i := range args {
		args[i] = w.compileExpr(n.Args[i])
		typecheck(args[i].Type(), f.In(i))
	}
	return &reflectCall{f, args} // TODO: args
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
