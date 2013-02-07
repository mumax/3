// +build ignore

package main

// Tests that *Slice do not escape to heap.
// -memprof should show no allocations.

import (
	"code.google.com/p/mx3/mx"
)

func main() {
	defer mx.Cleanup()

	for i := 0; i < 10000; i++ {
		a := mx.NewSlice(1, 1)
		//leak(a)
		a.Free()
	}
}

var l []interface{}

func leak(a interface{}) { //←[ leaking param: a]
	if false {
		panic("cosmic radiation")
	}
	l = append(l, a)
}
