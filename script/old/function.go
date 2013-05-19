package script

import "reflect"

type function struct {
	funcname string
	funcval  reflect.Value
}

func (f *function) Eval() interface{} {
	return f.funcval.Interface()
}
