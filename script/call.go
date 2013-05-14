package script

import (
	"fmt"
	"reflect"
)

type call struct {
	function
	args []Expr
}

func (p *Parser) newCall(name string, args []Expr) *call {
	f, ok := p.get(name).(*function)
	if !ok {
		panic(fmt.Errorf("line %v: cannot call %v (type %v)", p.Position, name, reflect.TypeOf(p.get(name))))
	}
	functyp := f.funcval.Type()
	if !functyp.IsVariadic() && functyp.NumIn() != len(args) {
		panic(fmt.Errorf("line %v: %v needs %v arguments, have %v", p.Line, name, functyp.NumIn(), len(args)))
	}
	return &call{function{name, f.funcval}, args}
}

func (e *call) Eval() interface{} {
	argv := List(e.args).Eval().([]interface{})
	if len(argv) != e.funcval.Type().NumIn() {
		panic(fmt.Errorf("call %v with wrong number of arguments: want: %v, have: %v", e.funcname, e.funcval.Type().NumIn(), len(argv)))
	}
	args := make([]reflect.Value, len(argv))
	for i := range args {
		args[i] = convertArg(argv[i], e.funcval.Type().In(i))
	}
	ret := e.function.funcval.Call(args)
	if len(ret) == 0 {
		return nil
	}
	if len(ret) == 1 {
		return ret[0].Interface()
	}
	return ret // multiple return values still returned as []reflect.Value
}

func convertArg(v interface{}, typ reflect.Type) reflect.Value {
	switch typ.Kind() {
	case reflect.Int:
		return reflect.ValueOf(cint(v.(float64)))
	case reflect.Float32:
		return reflect.ValueOf(float32(v.(float64)))
	default:
		return reflect.ValueOf(v) // do not convert
	}
}

func (e *call) String() string {
	str := fmt.Sprint(e.funcname, "( ") // todo: addr2line
	for _, a := range e.args {
		str += fmt.Sprint(a, " ")
	}
	str += ")"
	return str
}

// safe conversion from float to integer.
func cint(f float64) int {
	i := int(f)
	if float64(i) != f {
		panic(fmt.Errorf("need integer, have: %v", f))
	}
	return i
}
