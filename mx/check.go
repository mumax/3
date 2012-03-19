package mx

import (
	"fmt"
)

func CheckUnits(a, b string) {
	if a != b {
		Panic("unit mismatch:", a, b)
	}
}

func CheckNComp(a, b int) {
	if a != b {
		Panic("components mismatch:", a, b)
	}
}

func Panic(msg ...interface{}) {
	panic(fmt.Sprint(msg))
}
