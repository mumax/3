package script

import (
	"fmt"
)

type expr interface {
	eval() interface{}
}

type nop struct{}

func (e *nop) eval() interface{} {
	return nil
}

type variable struct {
	name string
}

func (e *variable) eval() interface{} {
	return e.name
}

type call struct {
	funcname string
	args     []expr
}

func (e *call) eval() interface{} {
	str := fmt.Sprint(e.funcname, "(")
	for _, a := range e.args {
		str += fmt.Sprint(a.eval(), " ")
	}
	str += ")"
	return str
}

//func (e*call)exec(){
//	fmt.Println(e.eval()) // don't use result
//}

type assign struct {
	left  string
	right expr
}

func (e *assign) eval() interface{} {
	fmt.Println(e.left, "=", e.right)
	return nil
}

type num float64

func (n num) eval() interface{} {
	return float64(n)
}
