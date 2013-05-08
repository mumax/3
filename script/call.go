package script

import (
	"fmt"
	"reflect"
)

type call struct {
	funcname string
	funcval  reflect.Value
	args     []Expr
}

func (p *Parser) newCall(name string, args []Expr) *call {
	funcval := p.getfunc(name)
	functyp := funcval.Type()
	if !functyp.IsVariadic() && functyp.NumIn() != len(args) {
		panic(fmt.Errorf("line %v: %v needs %v arguments, have %v", p.scan.Pos().Line, name, functyp.NumIn(), len(args)))
	}
	return &call{name, funcval, args}
}

func (e *call) Eval() interface{} {
	argv := List(e.args).Eval().([]interface{})
	args := make([]reflect.Value, len(argv))
	for i := range args {
		args[i] = reflect.ValueOf(argv[i])
	}
	return e.funcval.Call(args)
}

func (e *call) String() string {
	str := fmt.Sprint(e.funcname, "( ") // todo: addr2line
	for _, a := range e.args {
		str += fmt.Sprint(a, " ")
	}
	str += ")"
	return str
}
