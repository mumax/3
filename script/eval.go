package script

import (
	"fmt"
)

type expr interface {
	eval() interface{}
	String() string
}

type nop struct{}

func (e *nop) eval() interface{} {
	return nil
}

func (e *nop) String() string {
	return ";"
}

type variable struct {
	name string
}

func (e *variable) eval() interface{} {
	return e.name
}

func (e *variable) String() string {
	return e.name
}

type call struct {
	funcname string
	args     []expr
}

func (e *call) eval() interface{} {
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

type assign struct {
	left  string
	right expr
}

func (e *assign) eval() interface{} {
	fmt.Println(e.left, "=", e.right)
	return nil
}

func (e *assign) String() string {
	return fmt.Sprint(e.left, "=", e.right)
}

type num float64

func (n num) eval() interface{} {
	return float64(n)
}

func (n num) String() string {
	return fmt.Sprint(n.eval())
}
