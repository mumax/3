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
	args := make([]reflect.Value, len(argv))
	for i := range args {
		args[i] = reflect.ValueOf(argv[i])
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

func (e *call) String() string {
	str := fmt.Sprint(e.funcname, "( ") // todo: addr2line
	for _, a := range e.args {
		str += fmt.Sprint(a, " ")
	}
	str += ")"
	return str
}
