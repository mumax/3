package script

import (
	"reflect"
)

// left-hand value in (single) assign statement
type lvalue interface {
	expr
	Set(interface{})
}

// general lvalue implementation using reflect
type reflectLvalue struct {
	addr reflect.Value
}

// lhs must be settable, e.g. address of something
func newReflectLvalue(lhs interface{}) lvalue {
	return &reflectLvalue{reflect.ValueOf(lhs).Elem()}
}

// implements lvalue
func (l *reflectLvalue) Set(rvalue interface{}) {
	l.addr.Set(reflect.ValueOf(rvalue))
}

func (l *reflectLvalue) Eval() []interface{} {
	return []interface{}{l.addr.Interface()}
}

func (l *reflectLvalue) NumOut() int {
	return 1
}
