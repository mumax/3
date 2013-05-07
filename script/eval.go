package script

import (
	"fmt"
)

func nop() interface{} { return "nop" }

func makeVariable(name string)fn{
	return func()interface{}{return name}
}

func call(function string, args []fn) interface{} {
	str := fmt.Sprint(function, "(")
	for _, f := range args {
		str += fmt.Sprint(f(), " ")
	}
	str += ")"
	return str
}
