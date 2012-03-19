package mx

import (
	"fmt"
)

func CheckUnits(a, b string) { //←[ leaking param: a  leaking param: b]
	if a != b {
		Panic("unit mismatch:", a, b) //←[ ... argument escapes to heap]
	}
}

func CheckNComp(a, b int) {
	if a != b {
		Panic("components mismatch:", a, b) //←[ ... argument escapes to heap]
	}
}

func Panic(msg ...interface{}) { //←[ leaking param: msg]
	panic(fmt.Sprint(msg)) //←[ Panic ... argument does not escape]
}
