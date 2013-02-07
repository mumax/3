// +build ignore

package main

// Tests that kernel launches through PTX wrappers do not produce garbage.
// -memprof should show no allocations.

import (
	"code.google.com/p/mx3/mx"
)

func main() {
	defer mx.Cleanup()

	a := mx.NewSlice(1, 100)
	for i := 0; i < 10000; i++ {
		mx.Sum(a)
	}
}
