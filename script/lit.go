package script

import (
	"go/ast"
	"go/token"
	"reflect"
	"strconv"
)

// compiles a basic literal, like numbers and strings
func (w *World) compileBasicLit(n *ast.BasicLit) expr {
	switch n.Kind {
	default:
		panic(err("not allowed:", n.Value, "(", typ(n), ")"))
	case token.INT, token.FLOAT:
		return floatLit(parseFloat(n.Value))
	}
}

type floatLit float64

func (l floatLit) Eval() interface{}  { return float64(l) }
func (l floatLit) Type() reflect.Type { return float64_t }

func parseFloat(str string) float64 {
	v, err := strconv.ParseFloat(str, 64)
	if err != nil {
		panic("internal error") // we were sure it was a number...
	}
	return v
}

// TODO: rename relfectFunc
type funcLit struct {
	fn reflect.Value
}

func newFuncLit(fn interface{}) *funcLit {
	val := reflect.ValueOf(fn)
	if val.Type().Kind() != reflect.Func {
		panic(err("not a function:", fn))
	}
	if val.Type().NumOut() > 1 {
		panic(err("multiple return values not allowed:", fn))
	}
	return &funcLit{val}
}

func (f *funcLit) call(args []interface{}) interface{} {
	ret := f.fn.Call(argv(args))
	assert(len(ret) <= 1)
	if len(ret) == 0 {
		return nil
	} else {
		return ret[0]
	}
}

// convert []interface{} to []reflect.Value
func argv(iface []interface{}) []reflect.Value {
	v := make([]reflect.Value, len(iface))
	for i := range iface {
		v[i] = reflect.ValueOf(iface[i])
	}
	return v
}

func (f *funcLit) Type() reflect.Type {
	switch f.fn.Type().NumOut() {
	case 0:
		return nil // "void"
	case 1:
		return f.fn.Type().Out(0)
	default:
		panic("bug: multiple return values not allowed")
	}
}

func (f *funcLit) NumIn() int {
	return f.fn.Type().NumIn()
}

func (f *funcLit) In(i int) reflect.Type {
	return f.fn.Type().In(i)
}

func (f *funcLit) Eval() interface{} {
	return f.fn.Interface()
}
