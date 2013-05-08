package script

import (
	"fmt"
)

type nop struct{}

func (e *nop) Eval() interface{} {
	return nil
}

func (e *nop) String() string {
	return ";"
}

type call struct {
	funcname string
	args     []Expr
}

func (e *call) Eval() interface{} {
	return nil
}

func (e *call) String() string {
	str := fmt.Sprint(e.funcname, "( ")
	for _, a := range e.args {
		str += fmt.Sprint(a, " ")
	}
	str += ")"
	return str
}

type num float64

func (n num) Eval() interface{} {
	return float64(n)
}

func (n num) String() string {
	return fmt.Sprint(n.Eval())
}
