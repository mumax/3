package script

import (
	"fmt"
	"reflect"
)

type reflectFunc struct {
	fn reflect.Value
}

func newReflectFunc(fn interface{}) *reflectFunc {
	val := reflect.ValueOf(fn)
	if val.Type().Kind() != reflect.Func {
		panic(fmt.Errorf("not a function: %v", val.Type()))
	}
	if val.Type().NumOut() > 1 {
		panic(fmt.Errorf("multiple return values not allowed: %v", val.Type()))
	}
	return &reflectFunc{val}
}

func (f *reflectFunc) call(args []interface{}) interface{} {
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

func (f *reflectFunc) Type() reflect.Type {
	switch f.fn.Type().NumOut() {
	case 0:
		return nil // "void"
	case 1:
		return f.fn.Type().Out(0)
	default:
		panic("bug: multiple return values not allowed")
	}
}

func (f *reflectFunc) NumIn() int {
	return f.fn.Type().NumIn()
}

func (f *reflectFunc) In(i int) reflect.Type {
	return f.fn.Type().In(i)
}

func (f *reflectFunc) Eval() interface{} {
	return f.fn.Interface()
}
