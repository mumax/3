package script

import (
	"fmt"
	"reflect"
)

type function struct {
	reflect.Value
}

func newFunction(fn interface{}) *function {
	val := reflect.ValueOf(fn)
	if val.Type().Kind() != reflect.Func {
		panic(fmt.Errorf("not a function: %v", val.Type()))
	}
	if val.Type().NumOut() > 1 {
		panic(fmt.Errorf("multiple return values not allowed: %v", val.Type()))
	}
	return &function{val}
}

// type of the function itself (when not called)
func (f *function) Type() reflect.Type    { return f.Value.Type() }
func (f *function) NumIn() int            { return f.Type().NumIn() }
func (f *function) In(i int) reflect.Type { return f.Type().In(i) }
func (f *function) Eval() interface{}     { return f.Value.Interface() }
func (f *function) Child() []Expr         { return nil }
func (f *function) Fix() Expr             { return f }
