package cgo

// #include "test.h"
import "C"

func Test() int {
	return int(C.test())
}
