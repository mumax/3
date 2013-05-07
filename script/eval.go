package script

import (
	"fmt"
)

func nop() interface{} { return "nop" }

func eval(ident string, args []fn) interface{} {
	str := fmt.Sprint(ident, "(")
	for _, f := range args {
		str += fmt.Sprint(f(), " ")
	}
	str += ")"
	return str
}
