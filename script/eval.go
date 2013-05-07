package script

import (
	"fmt"
)

type node interface {
	eval() interface{}
}

type nop struct{}

func (n *nop) eval() interface{} {
	return nil
}

type variable struct{ name string }

func (n *variable) eval() interface{} { return n.name }

type call struct {
	funcname string
	args     []node
}

func (n *call) eval() interface{} {
	str := fmt.Sprint(n.funcname, "(")
	for _, f := range n.args {
		str += fmt.Sprint(f.eval(), " ")
	}
	str += ")"
	return str
}
